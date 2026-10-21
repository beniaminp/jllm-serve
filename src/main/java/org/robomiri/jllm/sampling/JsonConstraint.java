package org.robomiri.jllm.sampling;

import org.robomiri.jllm.json.Json;
import org.robomiri.jllm.tokenizer.Tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;

/**
 * Constrained decoding for JSON output (OpenAI {@code response_format}): a token is allowed only when the output
 * with its bytes appended is still the prefix of a JSON value that matches the {@link Schema}. The check runs a
 * byte-level pushdown automaton (a stack of frames: object, array, string, number, literal) on a copy of the
 * current state, so testing a candidate costs a copy of a few frames plus one step per byte; when the sampler has
 * to look beyond its top candidates the whole allowed set is computed with the vocabulary trie
 * ({@link ByteConstraint}). Once the value
 * is complete nothing more is allowed ({@link #mustEnd}); a top-level number may end at any digit ({@link #canEnd}).
 *
 * <p>Whitespace between tokens is allowed but capped at one newline and {@value #MAX_WS} bytes in a row (enough
 * for pretty-printing): under a constraint a model that would rather write prose otherwise happily emits blank
 * lines forever. Strings must be valid UTF-8 without raw control
 * characters; escapes are checked (a {@code u} escape needs four hex digits).
 */
public final class JsonConstraint extends ByteConstraint {

    /** Longest run of whitespace allowed between JSON tokens. */
    static final int MAX_WS = 64;

    // value kinds, by the first byte of a value
    private static final int K_OBJECT = 1, K_ARRAY = 2, K_STRING = 4, K_NUMBER = 8, K_BOOLEAN = 16, K_NULL = 32;

    /**
     * A compiled JSON schema, restricted to what the automaton enforces: {@code type} (object, array, string,
     * number, integer, boolean, null, or a list of them), {@code properties}, {@code required},
     * {@code additionalProperties} (boolean or a schema), {@code items}, {@code enum}, {@code const},
     * {@code anyOf} / {@code oneOf} whose alternatives start differently (e.g. a string or null: the first byte of
     * the value picks the alternative), local {@code $ref} ({@code #}, {@code #/$defs/...},
     * {@code #/definitions/...}, recursion allowed), {@code minItems} / {@code maxItems},
     * {@code minLength} / {@code maxLength} and OpenAPI's {@code nullable}. Annotations ({@code title},
     * {@code description}, {@code default}, {@code examples}, {@code format}, ...) are ignored; any other keyword is
     * rejected by {@link #compile}, so that a schema is never silently enforced only in part.
     */
    public static final class Schema {
        static final int T_OBJECT = 1, T_ARRAY = 2, T_STRING = 4, T_NUMBER = 8, T_INTEGER = 16, T_BOOLEAN = 32,
                T_NULL = 64, T_ANY = 127;

        int types = T_ANY;
        /** Property schemas by encoded name (the bytes between the quotes, as the model must write them). */
        Map<String, Schema> properties = Map.of();
        /** Encoded names a closed object may use: its properties and required keys. */
        byte[][] keyNames = new byte[0][];
        String[] required = new String[0];
        boolean additional = true;
        Schema additionalSchema; // null = anything
        Schema items; // null = anything
        int minItems;
        int maxItems = Integer.MAX_VALUE;
        int minLength;
        int maxLength = Integer.MAX_VALUE;
        byte[][] literals; // enum / const in compact serialization, null = no enum
        Schema[] alternatives; // anyOf / oneOf
        Schema target; // $ref

        private static final Set<String> SUPPORTED = Set.of("type", "properties", "required", "additionalProperties",
                "items", "enum", "const", "anyOf", "oneOf", "$ref", "$defs", "definitions", "minItems", "maxItems",
                "minLength", "maxLength", "nullable");
        private static final Set<String> IGNORED = Set.of("title", "description", "default", "examples", "$schema",
                "$id", "$comment", "format", "deprecated", "readOnly", "writeOnly", "strict", "name");

        /** Any JSON value. */
        public static Schema any() {
            return new Schema();
        }

        /** Any JSON object ({@code response_format: json_object}). */
        public static Schema anyObject() {
            Schema s = new Schema();
            s.types = T_OBJECT;
            return s;
        }

        /** @throws IllegalArgumentException for unsupported keywords or a malformed schema, naming the problem */
        public static Schema compile(Map<String, Object> schema) {
            return compile(schema, false);
        }

        /**
         * @param lenient for schemas the client did not write for us (tool parameters): a keyword the automaton
         *                cannot enforce ({@code pattern}, {@code minimum}, {@code allOf}, ...) is ignored instead of
         *                rejected, as are tuple {@code items}, non-local {@code $ref}s and unknown types (they allow
         *                anything), and {@code anyOf} / {@code oneOf} alternatives of the same JSON type are merged
         *                into that type: the constraint is then looser than the schema, never stricter
         * @throws IllegalArgumentException for a malformed schema (or, when not lenient, an unsupported keyword)
         */
        public static Schema compile(Map<String, Object> schema, boolean lenient) {
            Map<String, Schema> refs = new HashMap<>();
            List<Runnable> checks = new ArrayList<>();
            Schema root = new Schema(); // registered first, so that a "$ref": "#" inside finds it
            refs.put("#", root);
            Compiler compiler = new Compiler(schema, refs, checks, lenient);
            root.target = compiler.compile(schema, "#");
            for (Runnable c : checks) { // with every $ref filled in
                c.run();
            }
            return root;
        }

        /** One compilation: the root document, the $ref targets so far, the checks to run at the end. */
        private record Compiler(Map<String, Object> root, Map<String, Schema> refs, List<Runnable> checks,
                                boolean lenient) {
        private Schema compile(Object node, String path) {
            if (Boolean.TRUE.equals(node) || (lenient && !(node instanceof Map<?, ?>))) {
                return any();
            }
            if (!(node instanceof Map<?, ?>)) {
                throw new IllegalArgumentException("JSON schema at " + path + " must be an object");
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> m = (Map<String, Object>) node;
            for (String k : m.keySet()) {
                if (!SUPPORTED.contains(k) && !IGNORED.contains(k) && !lenient) {
                    throw new IllegalArgumentException("JSON schema keyword '" + k + "' (at " + path + ") is not"
                            + " supported; supported: " + String.join(", ", new java.util.TreeSet<>(SUPPORTED)));
                }
            }
            Schema s = new Schema();
            if (m.get("$ref") instanceof String ref) {
                Schema t = refs.get(ref);
                if (t == null) {
                    Object target;
                    try {
                        target = pointer(root, ref);
                    } catch (IllegalArgumentException e) {
                        if (lenient) {
                            return any();
                        }
                        throw e;
                    }
                    t = new Schema(); // placeholder first: a recursive reference finds it
                    refs.put(ref, t);
                    t.target = compile(target, ref);
                }
                return t;
            } else if (m.containsKey("$ref")) {
                throw new IllegalArgumentException("'$ref' at " + path + " must be a string");
            }
            boolean typed = false;
            Object type = m.get("type");
            if (type != null) {
                s.types = 0;
                for (Object t : type instanceof List<?> l ? l : List.of(type)) {
                    s.types |= switch (String.valueOf(t)) {
                        case "object" -> T_OBJECT;
                        case "array" -> T_ARRAY;
                        case "string" -> T_STRING;
                        case "number" -> T_NUMBER;
                        case "integer" -> T_INTEGER;
                        case "boolean" -> T_BOOLEAN;
                        case "null" -> T_NULL;
                        default -> {
                            if (lenient) {
                                yield T_ANY;
                            }
                            throw new IllegalArgumentException("unknown JSON schema type '" + t + "' at " + path);
                        }
                    };
                }
                typed = true;
            }
            if (m.containsKey("enum") || m.containsKey("const")) {
                List<?> values = m.containsKey("const") ? java.util.Collections.singletonList(m.get("const")) : asList(m.get("enum"), "enum", path);
                if (values.isEmpty()) {
                    throw new IllegalArgumentException("'enum' at " + path + " is empty");
                }
                s.literals = new byte[values.size()][];
                for (int i = 0; i < values.size(); i++) {
                    s.literals[i] = Json.write(values.get(i)).getBytes(StandardCharsets.UTF_8);
                }
                typed = true;
            }
            for (String key : new String[]{"anyOf", "oneOf"}) {
                if (m.containsKey(key)) {
                    List<?> alts = asList(m.get(key), key, path);
                    if (alts.isEmpty()) {
                        throw new IllegalArgumentException("'" + key + "' at " + path + " is empty");
                    }
                    Schema[] alt = new Schema[alts.size()];
                    for (int i = 0; i < alts.size(); i++) {
                        alt[i] = compile(alts.get(i), path + "/" + key + "/" + i);
                    }
                    s.alternatives = alt;
                    boolean explicit = type != null;
                    checks.add(() -> { // (an alternative may be a $ref that is not filled in yet)
                        int seen = 0;
                        for (Schema a : alt) {
                            int k = a.kinds();
                            if ((k & seen) != 0 && lenient) { // merge: any value of the alternatives' types
                                int all = 0;
                                for (Schema b : alt) {
                                    all |= b.kinds();
                                }
                                int t = typesOf(all);
                                s.alternatives = null;
                                s.types = explicit && (s.types & t) != 0 ? s.types & t : t;
                                return;
                            }
                            if ((k & seen) != 0) {
                                throw new IllegalArgumentException("'" + key + "' at " + path + ": the alternatives"
                                        + " must differ in JSON type (object, array, string, number, boolean, null),"
                                        + " e.g. a string or null; overlapping alternatives are not supported");
                            }
                            seen |= k;
                        }
                    });
                    typed = true;
                }
            }
            if (Boolean.TRUE.equals(m.get("nullable"))) {
                s.types |= T_NULL;
            }
            Map<String, Object> props = Json.map(m, "properties");
            if (m.containsKey("properties") && props == null) {
                throw new IllegalArgumentException("'properties' at " + path + " must be an object");
            }
            Set<String> names = new LinkedHashSet<>();
            if (props != null) {
                Map<String, Schema> p = new LinkedHashMap<>();
                for (Map.Entry<String, Object> e : props.entrySet()) {
                    p.put(encode(e.getKey()), compile(e.getValue(), path + "/properties/" + e.getKey()));
                    names.add(e.getKey());
                }
                s.properties = p;
            }
            if (m.containsKey("required")) {
                List<?> req = asList(m.get("required"), "required", path);
                s.required = new String[req.size()];
                for (int i = 0; i < req.size(); i++) {
                    s.required[i] = encode(String.valueOf(req.get(i)));
                    names.add(String.valueOf(req.get(i)));
                }
            }
            s.keyNames = new byte[names.size()][];
            int i = 0;
            for (String n : names) {
                s.keyNames[i++] = encode(n).getBytes(StandardCharsets.UTF_8);
            }
            Object add = m.get("additionalProperties");
            if (add instanceof Boolean b) {
                s.additional = b;
            } else if (add != null) {
                s.additionalSchema = compile(add, path + "/additionalProperties");
            }
            if (m.containsKey("items")) {
                if (m.get("items") instanceof List<?>) {
                    if (!lenient) {
                        throw new IllegalArgumentException("tuple 'items' (an array of schemas) at " + path + " is not supported");
                    }
                } else {
                    s.items = compile(m.get("items"), path + "/items");
                }
            }
            s.minItems = count(m, "minItems", 0, path);
            s.maxItems = count(m, "maxItems", Integer.MAX_VALUE, path);
            s.minLength = count(m, "minLength", 0, path);
            s.maxLength = count(m, "maxLength", Integer.MAX_VALUE, path);
            if (!typed) { // no type: infer it from the keywords, as people write {"properties": ...} for an object
                if (props != null || m.containsKey("required") || add != null) {
                    s.types = T_OBJECT;
                } else if (m.containsKey("items") || m.containsKey("minItems") || m.containsKey("maxItems")) {
                    s.types = T_ARRAY;
                } else if (m.containsKey("minLength") || m.containsKey("maxLength")) {
                    s.types = T_STRING;
                }
            }
            if (s.types == 0) {
                throw new IllegalArgumentException("'type' at " + path + " is empty");
            }
            return s;
        }
        }

        /** The schema types of value kinds (a number kind allows any number). */
        private static int typesOf(int kinds) {
            return ((kinds & K_OBJECT) != 0 ? T_OBJECT : 0) | ((kinds & K_ARRAY) != 0 ? T_ARRAY : 0)
                    | ((kinds & K_STRING) != 0 ? T_STRING : 0) | ((kinds & K_NUMBER) != 0 ? T_NUMBER | T_INTEGER : 0)
                    | ((kinds & K_BOOLEAN) != 0 ? T_BOOLEAN : 0) | ((kinds & K_NULL) != 0 ? T_NULL : 0);
        }

        private static List<?> asList(Object o, String key, String path) {
            if (!(o instanceof List<?> l)) {
                throw new IllegalArgumentException("'" + key + "' at " + path + " must be an array");
            }
            return l;
        }

        private static int count(Map<String, Object> m, String key, int def, String path) {
            Object v = m.get(key);
            if (v == null) {
                return def;
            }
            if (!(v instanceof Number n) || n.doubleValue() < 0 || n.doubleValue() != Math.rint(n.doubleValue())) {
                throw new IllegalArgumentException("'" + key + "' at " + path + " must be a non-negative integer");
            }
            return (int) Math.min(Integer.MAX_VALUE, n.longValue());
        }

        /** Resolves a local reference: {@code #}, or a JSON pointer such as {@code #/$defs/Address}. */
        private static Object pointer(Map<String, Object> root, String ref) {
            if (!ref.startsWith("#")) {
                throw new IllegalArgumentException("only local '$ref's (starting with #) are supported: " + ref);
            }
            Object node = root;
            for (String part : ref.substring(1).split("/")) {
                if (part.isEmpty()) {
                    continue;
                }
                String key = part.replace("~1", "/").replace("~0", "~");
                node = node instanceof Map<?, ?> mm ? mm.get(key) : null;
                if (node == null) {
                    throw new IllegalArgumentException("'$ref' " + ref + " points nowhere");
                }
            }
            return node;
        }

        /** Follows {@code $ref}s. */
        Schema resolve() {
            Schema s = this;
            for (int i = 0; s.target != null; i++) {
                if (i > 64) {
                    throw new IllegalStateException("a $ref cycle without content");
                }
                s = s.target;
            }
            return s;
        }

        /** The value kinds this schema can start with. */
        int kinds() {
            Schema s = resolve();
            if (s.literals != null) {
                int k = 0;
                for (byte[] l : s.literals) {
                    k |= kindOf(l[0] & 0xff);
                }
                return k;
            }
            if (s.alternatives != null) {
                int k = 0;
                for (Schema a : s.alternatives) {
                    k |= a.kinds();
                }
                return k;
            }
            int t = s.types;
            return ((t & T_OBJECT) != 0 ? K_OBJECT : 0) | ((t & T_ARRAY) != 0 ? K_ARRAY : 0)
                    | ((t & T_STRING) != 0 ? K_STRING : 0) | ((t & (T_NUMBER | T_INTEGER)) != 0 ? K_NUMBER : 0)
                    | ((t & T_BOOLEAN) != 0 ? K_BOOLEAN : 0) | ((t & T_NULL) != 0 ? K_NULL : 0);
        }

        /** The (resolved) schema a value starting with kind {@code k} is checked against, or null if none allows it. */
        Schema forKind(int k) {
            Schema s = resolve();
            if (s.alternatives != null) {
                for (Schema a : s.alternatives) {
                    Schema r = a.forKind(k);
                    if (r != null) {
                        return r;
                    }
                }
                return null;
            }
            return (s.kinds() & k) != 0 ? s : null;
        }
    }

    // ------------------------------------------------------------------ the automaton

    private static final int F_VALUE = 0, F_OBJECT = 1, F_ARRAY = 2, F_STRING = 3, F_NUMBER = 4, F_LITERAL = 5;
    /** Object / array phases: after the opening bracket, after a comma, after a key, inside a value, after a value. */
    private static final int P_OPEN = 0, P_NEXT = 1, P_COLON = 2, P_VALUE = 3, P_AFTER = 4;
    /** Number states (JSON grammar). */
    private static final int N_START = 0, N_MINUS = 1, N_ZERO = 2, N_INT = 3, N_DOT = 4, N_FRAC = 5, N_E = 6,
            N_ESIGN = 7, N_EXP = 8;
    private static final int OK = 0, FAIL = 1, AGAIN = 2;
    private static final String[] NONE = new String[0];
    private static final byte[][] TRUE_FALSE = {"true".getBytes(StandardCharsets.US_ASCII), "false".getBytes(StandardCharsets.US_ASCII)};
    private static final byte[][] NULL_LIT = {"null".getBytes(StandardCharsets.US_ASCII)};
    private static final Schema ANY = Schema.any();

    private static final class Frame {
        int kind;
        Schema schema;
        int phase;
        String[] seen = NONE; // object: the encoded keys so far
        String key; // object: the key whose value comes next
        int count; // array: items so far; string: characters so far
        int esc; // string: 0, -1 right after a backslash, k > 0 = hex digits still due
        int cont; // string: UTF-8 continuation bytes still due
        Frame owner; // key string: its object
        byte[] buf; // key string: the bytes so far
        int len;
        int num; // number state
        boolean integer;
        byte[][] lits; // literal: the candidates still matching
        int pos;

        Frame copy() {
            Frame f = new Frame();
            f.kind = kind;
            f.schema = schema;
            f.phase = phase;
            f.seen = seen;
            f.key = key;
            f.count = count;
            f.esc = esc;
            f.cont = cont;
            f.owner = owner;
            f.buf = buf == null ? null : buf.clone();
            f.len = len;
            f.num = num;
            f.integer = integer;
            f.lits = lits;
            f.pos = pos;
            return f;
        }
    }

    /**
     * The automaton's state: a stack of frames. In <b>canonical</b> mode (tool-call arguments) the whitespace is
     * fixed to what Python's {@code json.dumps} writes, the format the chat templates render tool calls in: one
     * space after every {@code :} and {@code ,} between tokens, none anywhere else ({@code {"a": 1, "b": [1, 2]}}).
     * The model loses nothing (it was trained on that layout) and the structure between values becomes a single
     * possible byte sequence, which jump-forward can append without sampling.
     */
    static final class State implements ByteConstraint.State {
        Frame[] stack = new Frame[8];
        int depth;
        int ws;
        int nl; // newlines in the current whitespace run
        final boolean canonical;
        boolean needSpace; // canonical: the byte after a structural ':' or ',' must be a space

        State(boolean canonical) {
            this.canonical = canonical;
        }

        /** The start state of a value of {@code schema}. */
        static State start(Schema schema, boolean canonical) {
            State s = new State(canonical);
            s.push(s.value(schema));
            return s;
        }

        @Override
        public boolean canEnd() {
            return canEndValue();
        }

        @Override
        public boolean done() {
            return depth == 0;
        }

        /** Inside a string value, or a key of an object that takes any key. */
        @Override
        public boolean wide() {
            if (depth == 0) {
                return false;
            }
            Frame f = stack[depth - 1];
            return f.kind == F_STRING && (f.owner == null || f.owner.schema.additional);
        }

        /**
         * The whole stack: kinds, schemas (by identity), phases, keys seen, partial key bytes, counters. Literal
         * frames (enum matching in progress) are not memoized.
         */
        @Override
        public Object key() {
            List<Object> k = new ArrayList<>(depth * 6 + 4);
            k.add(canonical);
            k.add(needSpace);
            k.add(ws);
            k.add(nl);
            for (int i = 0; i < depth; i++) {
                Frame f = stack[i];
                if (f.kind == F_LITERAL) {
                    return null;
                }
                k.add(f.kind);
                k.add(new Ident(f.schema));
                k.add(f.phase);
                k.add(List.of(f.seen));
                k.add(f.key == null ? "" : f.key);
                Schema fs = f.schema;
                boolean counted = fs != null && (f.kind == F_STRING && f.owner == null
                        ? fs.minLength > 0 || fs.maxLength < Integer.MAX_VALUE
                        : f.kind == F_ARRAY && (fs.minItems > 0 || fs.maxItems < Integer.MAX_VALUE));
                k.add(counted ? f.count : 0); // (a count without limits does not change what is allowed)
                k.add(f.esc * 8 + f.cont);
                k.add(f.buf == null ? "" : new String(f.buf, 0, f.len, StandardCharsets.ISO_8859_1));
                k.add(f.num * 2 + (f.integer ? 1 : 0));
            }
            return k;
        }

        @Override
        public State copy() {
            State s = new State(canonical);
            s.needSpace = needSpace;
            s.stack = new Frame[stack.length];
            for (int i = 0; i < depth; i++) {
                Frame f = stack[i].copy();
                s.stack[i] = f;
            }
            for (int i = 0; i < depth; i++) { // a key frame points at its object: re-point to the copy
                if (s.stack[i].owner != null) {
                    s.stack[i].owner = s.stack[i - 1];
                }
            }
            s.depth = depth;
            s.ws = ws;
            s.nl = nl;
            return s;
        }

        void push(Frame f) {
            if (depth == stack.length) {
                stack = java.util.Arrays.copyOf(stack, depth * 2);
            }
            stack[depth++] = f;
        }

        Frame value(Schema s) {
            Frame f = new Frame();
            f.kind = F_VALUE;
            f.schema = s;
            return f;
        }

        @Override
        public boolean feed(int b) {
            if (needSpace) {
                if (b != ' ') {
                    return false;
                }
                needSpace = false;
                return true;
            }
            while (true) {
                if (depth == 0) {
                    return false; // the value is complete: nothing may follow
                }
                Frame f = stack[depth - 1];
                int r = switch (f.kind) {
                    case F_VALUE -> startValue(f, b);
                    case F_OBJECT -> object(f, b);
                    case F_ARRAY -> array(f, b);
                    case F_STRING -> string(f, b);
                    case F_NUMBER -> number(f, b);
                    default -> literal(f, b);
                };
                if (r != AGAIN) {
                    return r == OK;
                }
            }
        }

        /** Whitespace between tokens: at most one newline and {@value #MAX_WS} bytes in a row (indentation fits). */
        private int whitespace(int b) {
            if (canonical) {
                return FAIL; // (the one space after ':' and ',' is taken in feed)
            }
            if (b == '\n' && ++nl > 1) {
                return FAIL;
            }
            return ++ws > MAX_WS ? FAIL : OK;
        }

        /** A value ended: its frame is gone, the container (if any) moves past it. */
        private void valueDone() {
            if (depth == 0) {
                return;
            }
            Frame p = stack[depth - 1];
            if (p.kind == F_ARRAY) {
                p.count++;
            }
            p.phase = P_AFTER;
        }

        private void pop() {
            stack[--depth] = null;
        }

        private int startValue(Frame f, int b) {
            if (isWs(b)) {
                return whitespace(b);
            }
            ws = nl = 0;
            Schema s = f.schema.resolve();
            byte[][] lits = s.literals;
            int k = kindOf(b);
            Schema a = s;
            if (lits == null) {
                if (k == 0 || (a = s.forKind(k)) == null) {
                    return FAIL;
                }
                lits = a.literals;
            }
            if (lits == null && k == K_BOOLEAN) {
                lits = TRUE_FALSE;
            } else if (lits == null && k == K_NULL) {
                lits = NULL_LIT;
            }
            if (lits != null) {
                List<byte[]> match = new ArrayList<>();
                for (byte[] l : lits) {
                    if ((l[0] & 0xff) == b) {
                        match.add(l);
                    }
                }
                if (match.isEmpty()) {
                    return FAIL;
                }
                f.kind = F_LITERAL;
                f.lits = match.toArray(new byte[0][]);
                f.pos = 1;
                settleLiteral(f);
                return OK;
            }
            f.schema = a;
            switch (k) {
                case K_OBJECT -> {
                    f.kind = F_OBJECT;
                    f.phase = P_OPEN;
                }
                case K_ARRAY -> {
                    f.kind = F_ARRAY;
                    f.phase = P_OPEN;
                }
                case K_STRING -> f.kind = F_STRING;
                default -> { // number
                    f.kind = F_NUMBER;
                    f.integer = (a.types & Schema.T_NUMBER) == 0;
                    f.num = N_START;
                    return number(f, b);
                }
            }
            return OK;
        }

        /** Pops a literal that is complete and cannot go on; one that could (enum 1 and 12) waits for the next byte. */
        private void settleLiteral(Frame f) {
            boolean complete = false;
            boolean longer = false;
            for (byte[] l : f.lits) {
                complete |= l.length == f.pos;
                longer |= l.length > f.pos;
            }
            if (complete && !longer) {
                pop();
                valueDone();
            }
        }

        private int literal(Frame f, int b) {
            List<byte[]> match = new ArrayList<>(f.lits.length);
            boolean complete = false;
            for (byte[] l : f.lits) {
                if (l.length > f.pos && (l[f.pos] & 0xff) == b) {
                    match.add(l);
                }
                complete |= l.length == f.pos;
            }
            if (!match.isEmpty()) {
                f.lits = match.toArray(new byte[0][]);
                f.pos++;
                settleLiteral(f);
                return OK;
            }
            if (complete) {
                pop();
                valueDone();
                return AGAIN;
            }
            return FAIL;
        }

        private int object(Frame f, int b) {
            Schema s = f.schema;
            if (f.phase == P_COLON) {
                if (isWs(b)) {
                    return whitespace(b);
                }
                ws = nl = 0;
                if (b != ':') {
                    return FAIL;
                }
                needSpace = canonical;
                f.phase = P_VALUE;
                Schema v = s.properties.get(f.key);
                push(value(v != null ? v : s.additionalSchema != null ? s.additionalSchema : ANY));
                return OK;
            }
            if (isWs(b)) {
                return whitespace(b);
            }
            ws = nl = 0;
            if (f.phase == P_AFTER) {
                if (b == ',') {
                    if (!canAddKey(f)) {
                        return FAIL;
                    }
                    needSpace = canonical;
                    f.phase = P_NEXT;
                    return OK;
                }
                return b == '}' ? close(f) : FAIL;
            }
            if (b == '"') {
                if (!canAddKey(f)) {
                    return FAIL;
                }
                Frame k = new Frame();
                k.kind = F_STRING;
                k.owner = f;
                k.buf = new byte[16];
                push(k);
                return OK;
            }
            return b == '}' && f.phase == P_OPEN ? close(f) : FAIL;
        }

        private int close(Frame f) {
            for (String r : f.schema.required) {
                if (!contains(f.seen, r)) {
                    return FAIL;
                }
            }
            pop();
            valueDone();
            return OK;
        }

        private static boolean canAddKey(Frame f) {
            if (f.schema.additional) {
                return true;
            }
            for (byte[] n : f.schema.keyNames) {
                if (!contains(f.seen, new String(n, StandardCharsets.UTF_8))) {
                    return true;
                }
            }
            return false;
        }

        private int array(Frame f, int b) {
            Schema s = f.schema;
            if (isWs(b)) {
                return whitespace(b);
            }
            if (f.phase == P_AFTER) {
                ws = nl = 0;
                if (b == ',') {
                    if (f.count >= s.maxItems) {
                        return FAIL;
                    }
                    needSpace = canonical;
                    f.phase = P_NEXT;
                    return OK;
                }
                return b == ']' ? closeArray(f) : FAIL;
            }
            if (b == ']' && f.phase == P_OPEN) {
                ws = nl = 0;
                return closeArray(f);
            }
            if (f.count >= s.maxItems) {
                return FAIL;
            }
            f.phase = P_VALUE;
            push(value(s.items != null ? s.items : ANY));
            return AGAIN; // the value frame takes this byte
        }

        private int closeArray(Frame f) {
            if (f.count < f.schema.minItems) {
                return FAIL;
            }
            pop();
            valueDone();
            return OK;
        }

        private int string(Frame f, int b) {
            boolean key = f.owner != null;
            if (f.esc < 0) { // right after a backslash
                if (b == 'u') {
                    f.esc = 4;
                } else if (b == '"' || b == '\\' || b == '/' || b == 'b' || b == 'f' || b == 'n' || b == 'r' || b == 't') {
                    f.esc = 0;
                } else {
                    return FAIL;
                }
                return key ? keyByte(f, b) : OK;
            }
            if (f.esc > 0) {
                if (!(b >= '0' && b <= '9' || b >= 'a' && b <= 'f' || b >= 'A' && b <= 'F')) {
                    return FAIL;
                }
                f.esc--;
                return key ? keyByte(f, b) : OK;
            }
            if (f.cont > 0) {
                if ((b & 0xC0) != 0x80) {
                    return FAIL;
                }
                f.cont--;
                return key ? keyByte(f, b) : OK;
            }
            if (b == '"') {
                return key ? keyDone(f) : endString(f);
            }
            if (b < 0x20) {
                return FAIL; // raw control characters must be escaped
            }
            if (b >= 0x80) {
                if (b >= 0xC2 && b <= 0xDF) {
                    f.cont = 1;
                } else if (b >= 0xE0 && b <= 0xEF) {
                    f.cont = 2;
                } else if (b >= 0xF0 && b <= 0xF4) {
                    f.cont = 3;
                } else {
                    return FAIL;
                }
            }
            if (b == '\\') {
                f.esc = -1;
            }
            if (!key && ++f.count > f.schema.maxLength) { // a character starts here (a lead byte or an escape)
                return FAIL;
            }
            return key ? keyByte(f, b) : OK;
        }

        private int endString(Frame f) {
            if (f.count < f.schema.minLength) {
                return FAIL;
            }
            pop();
            valueDone();
            return OK;
        }

        /** Appends a byte of an object key; a closed object only takes a prefix of one of its unused names. */
        private int keyByte(Frame f, int b) {
            if (f.len == f.buf.length) {
                f.buf = java.util.Arrays.copyOf(f.buf, f.len * 2);
            }
            f.buf[f.len++] = (byte) b;
            Schema s = f.owner.schema;
            if (s.additional) {
                return OK;
            }
            for (byte[] n : s.keyNames) {
                if (n.length >= f.len && java.util.Arrays.equals(n, 0, f.len, f.buf, 0, f.len)
                        && !contains(f.owner.seen, new String(n, StandardCharsets.UTF_8))) {
                    return OK;
                }
            }
            return FAIL;
        }

        private int keyDone(Frame f) {
            String name = new String(f.buf, 0, f.len, StandardCharsets.UTF_8);
            Frame o = f.owner;
            if (contains(o.seen, name)) {
                return FAIL; // no duplicate keys
            }
            if (!o.schema.additional) {
                boolean known = false;
                for (byte[] n : o.schema.keyNames) {
                    known |= n.length == f.len && java.util.Arrays.equals(n, 0, f.len, f.buf, 0, f.len);
                }
                if (!known) {
                    return FAIL;
                }
            }
            pop();
            String[] seen = java.util.Arrays.copyOf(o.seen, o.seen.length + 1);
            seen[o.seen.length] = name;
            o.seen = seen;
            o.key = name;
            o.phase = P_COLON;
            return OK;
        }

        private int number(Frame f, int b) {
            boolean digit = b >= '0' && b <= '9';
            boolean e = (b == 'e' || b == 'E') && !f.integer;
            int next = switch (f.num) {
                case N_START -> b == '-' ? N_MINUS : b == '0' ? N_ZERO : digit ? N_INT : -1;
                case N_MINUS -> b == '0' ? N_ZERO : digit ? N_INT : -1;
                case N_ZERO -> b == '.' && !f.integer ? N_DOT : e ? N_E : -1;
                case N_INT -> digit ? N_INT : b == '.' && !f.integer ? N_DOT : e ? N_E : -1;
                case N_DOT -> digit ? N_FRAC : -1;
                case N_FRAC -> digit ? N_FRAC : e ? N_E : -1;
                case N_E -> b == '+' || b == '-' ? N_ESIGN : digit ? N_EXP : -1;
                case N_ESIGN, N_EXP -> digit ? N_EXP : -1;
                default -> -1;
            };
            if (next >= 0) {
                f.num = next;
                return OK;
            }
            if (numberComplete(f)) { // the byte after a number belongs to its container
                pop();
                valueDone();
                return AGAIN;
            }
            return FAIL;
        }

        boolean canEndValue() {
            if (depth == 0) {
                return true;
            }
            if (depth > 1) {
                return false;
            }
            Frame f = stack[0];
            if (f.kind == F_NUMBER) {
                return numberComplete(f);
            }
            if (f.kind == F_LITERAL) {
                for (byte[] l : f.lits) {
                    if (l.length == f.pos) {
                        return true;
                    }
                }
            }
            return false;
        }
    }

    private static boolean numberComplete(Frame f) {
        return f.num == N_ZERO || f.num == N_INT || f.num == N_FRAC || f.num == N_EXP;
    }

    private static boolean contains(String[] a, String s) {
        for (String x : a) {
            if (x.equals(s)) {
                return true;
            }
        }
        return false;
    }

    private static boolean isWs(int b) {
        return b == ' ' || b == '\n' || b == '\t' || b == '\r';
    }

    private static int kindOf(int b) {
        return switch (b) {
            case '{' -> K_OBJECT;
            case '[' -> K_ARRAY;
            case '"' -> K_STRING;
            case 't', 'f' -> K_BOOLEAN;
            case 'n' -> K_NULL;
            default -> b == '-' || (b >= '0' && b <= '9') ? K_NUMBER : 0;
        };
    }

    /** A property name as the model must write it between the quotes: JSON escapes, otherwise raw UTF-8. */
    static String encode(String name) {
        String w = Json.write(name);
        return w.substring(1, w.length() - 1);
    }

    /** Identity wrapper, so that a memoization key compares schemas by identity. */
    private record Ident(Object o) {
        @Override
        public boolean equals(Object x) {
            return x instanceof Ident(Object other) && other == o;
        }

        @Override
        public int hashCode() {
            return System.identityHashCode(o);
        }
    }

    // ------------------------------------------------------------------ the constraint

    private static final Map<Tokenizer, byte[][]> TABLES = new WeakHashMap<>();

    /**
     * The bytes of every token, null for special tokens (which never belong in JSON) and empty tokens; computed
     * once per tokenizer.
     */
    public static byte[][] tokenTable(Tokenizer tok) {
        synchronized (TABLES) {
            return TABLES.computeIfAbsent(tok, t -> {
                byte[][] table = new byte[t.vocabSize()][];
                for (int i = 0; i < table.length; i++) {
                    byte[] b = t.tokenBytes(i);
                    table[i] = t.isSpecial(i) || b.length == 0 ? null : b;
                }
                return table;
            });
        }
    }

    /** @param tokens the bytes of each token id ({@link #tokenTable}) */
    public JsonConstraint(Schema schema, byte[][] tokens) {
        this(schema, tokens, false, null);
    }

    /**
     * @param canonical fixed {@code json.dumps} whitespace ({@link State}) instead of free whitespace
     * @param cache     memoized masks shared by the constraints of this schema, or null
     */
    JsonConstraint(Schema schema, byte[][] tokens, boolean canonical, MaskCache cache) {
        super(tokens, State.start(schema, canonical), cache);
    }

    private JsonConstraint(JsonConstraint o) {
        super(o);
    }

    @Override
    public JsonConstraint copy() {
        return new JsonConstraint(this);
    }

    /** For benchmarks: how many full allowed-token masks were computed by trie walks, and the nodes they visited. */
    public static long[] maskStats() {
        synchronized (ByteConstraint.class) {
            return new long[]{masksComputed, nodesVisited};
        }
    }

    /** A cache of allowed-token masks for the constraints of one schema ({@link Schema} keys are by identity). */
    public static Object newMaskCache() {
        return new MaskCache();
    }

    /** @param cache from {@link #newMaskCache}, shared by the constraints of {@code schema}; may be null */
    public JsonConstraint(Schema schema, byte[][] tokens, Object cache) {
        this(schema, tokens, false, (MaskCache) cache);
    }
}
