#!/usr/bin/env python3
"""Turns kernels.metal into C++ for the CPU emulator (metal_emu.h, runtime.cpp).

usage: transpile.py kernels.metal out.cpp [extra .metal sources...]

1. The source is run through the C preprocessor (macro-generated instantiations such as GG_INSTANTIATE expand) with
   <metal_stdlib> removed, then
2. MSL constructor calls float4(x) become mk_float4(x), `threadgroup` arrays declared inside functions become
   `static` (one copy shared by the fibers of the running threadgroup), and the `template [[host_name("...")]]`
   instantiations are removed and replaced by
3. one launcher per kernel that unpacks EmuArgs (buffers, threadgroup memory, thread position builtins) into the
   kernel's parameters, and an emu_register(name, launcher) call per instantiation or non-template kernel.
"""
import re
import subprocess
import sys

VEC = r'\b(float|half|uchar|char|ushort|short|int|uint|ulong)([2348])\('

BUILTIN = {
    'threadgroup_position_in_grid': 'a.tgPos',
    'thread_position_in_threadgroup': 'a.tid',
    'threads_per_threadgroup': 'a.tpg',
    'thread_position_in_grid': 'a.gid',
}
SCALAR_BUILTIN = {
    'simdgroup_index_in_threadgroup': 'a.sg',
    'thread_index_in_simdgroup': 'a.lane',
    'simdgroups_per_threadgroup': 'a.nsg',
    'thread_index_in_threadgroup': 'a.tidx',
}


def matching(s, i, open_c, close_c):
    """Index of the bracket closing the one at s[i]."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == open_c:
            depth += 1
        elif s[j] == close_c:
            depth -= 1
            if depth == 0:
                return j
    raise ValueError('unbalanced ' + open_c)


def split_top(s, sep=','):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '(<[':
            depth += 1
        elif ch in ')>]':
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def preprocess(paths):
    src = ''
    for p in paths:
        src += open(p).read() + '\n'
    src = src.replace('#include <metal_stdlib>', '').replace('using namespace metal;', '')
    src = '#define JLLM_ATTN_GROUPS 1\n' + src
    r = subprocess.run(['clang', '-E', '-P', '-x', 'c++', '-'], input=src, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(r.stderr)
    return r.stdout


def transpile(src):
    regs = []
    launchers = []
    # explicit instantiations: template [[host_name("a" "b")]] kernel <type> name<args>;
    inst = re.compile(r'template\s*\[\[\s*host_name\(((?:\s*"[^"]*")+)\s*\)\s*\]\]\s*kernel\s+(.*?);', re.S)

    def repl(m):
        name = ''.join(re.findall(r'"([^"]*)"', m.group(1)))
        decl = m.group(2).strip()
        k = re.search(r'(\w+)\s*(<[^<>]*>)\s*$', decl, re.S)
        regs.append((name, 'L_' + k.group(1) + k.group(2)))
        return ''
    src = inst.sub(repl, src)

    # kernels: [template <params>] kernel void name(params) {
    out = []
    pos = 0
    kern = re.compile(r'(template\s*<([^>]*)>\s*)?kernel\s+void\s+(\w+)\s*\(')
    for m in kern.finditer(src):
        open_i = m.end() - 1
        close_i = matching(src, open_i, '(', ')')
        params = src[open_i + 1:close_i]
        name = m.group(3)
        tparams = m.group(2)
        args = []
        for p in split_top(params):
            p = ' '.join(p.split())
            am = re.search(r'\[\[\s*(\w+)(?:\((\d+)\))?\s*\]\]', p)
            if not am:
                raise SystemExit('kernel %s: parameter without attribute: %s' % (name, p))
            decl = p[:am.start()].strip()
            dm = re.match(r'(.*?)(\w+)$', decl)
            ptype = dm.group(1).strip()
            attr, idx = am.group(1), am.group(2)
            if attr == 'buffer':
                if ptype.endswith('&'):
                    args.append('(*(%s *) a.buf[%s])' % (ptype[:-1].strip(), idx))
                else:
                    args.append('(%s) a.buf[%s]' % (ptype, idx))
            elif attr == 'threadgroup':
                args.append('(%s) a.tgmem' % ptype)
            elif attr in BUILTIN:
                args.append('emu_builtin<%s>(%s)' % (ptype, BUILTIN[attr]))
            elif attr in SCALAR_BUILTIN:
                args.append('(%s) %s' % (ptype, SCALAR_BUILTIN[attr]))
            else:
                raise SystemExit('kernel %s: unknown attribute %s' % (name, attr))
        call = ', '.join(args)
        if tparams:
            names = [re.match(r'.*?(\w+)\s*(=.*)?$', t.strip(), re.S).group(1) for t in split_top(tparams)]
            launchers.append('template <%s>\nstatic void L_%s(EmuArgs &a) { %s<%s>(%s); }\n'
                             % (tparams, name, name, ', '.join(names), call))
        else:
            launchers.append('static void L_%s(EmuArgs &a) { %s(%s); }\n' % (name, name, call))
            regs.append((name, 'L_' + name))

    # parameter attributes are only read above (the `threadgroup` macro would expand inside [[threadgroup(0)]])
    src = re.sub(r'\[\[\s*\w+(?:\(\d+\))?\s*\]\]', '', src)
    # MSL constructors -> mk_ functions; threadgroup arrays inside functions -> statics
    src = re.sub(VEC, r'mk_\1\2(', src)
    src = re.sub(r'(?m)^(\s+)threadgroup\s+(\w[\w\s]*?\w)\s+(\w+)\s*(\[[^;]*\]\s*;)', r'\1static threadgroup \2 \3\4', src)
    reg = ''.join('    emu_register("%s", &%s);\n' % (n, l) for n, l in regs)
    return ('#include "metal_emu.h"\n#pragma clang diagnostic ignored "-Wunknown-attributes"\n'
            + src + '\n// ---- launchers\n' + ''.join(launchers)
            + '\n__attribute__((constructor)) static void emu_register_all() {\n' + reg + '}\n', len(regs))


if __name__ == '__main__':
    code, n = transpile(preprocess([sys.argv[1]] + sys.argv[3:]))
    open(sys.argv[2], 'w').write(code)
    print('%d kernels' % n, file=sys.stderr)
