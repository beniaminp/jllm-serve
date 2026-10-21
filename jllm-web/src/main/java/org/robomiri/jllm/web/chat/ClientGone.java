package org.robomiri.jllm.web.chat;

import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletRequestWrapper;
import jakarta.servlet.http.HttpServletRequest;

import java.lang.reflect.Field;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;

/**
 * Whether the client of a request has closed its connection, so that a generation nobody waits for stops (docs/10
 * §10.11). Servlets offer no portable way to notice this without writing: a non-streaming request would keep its
 * slot for the whole generation, and a stream only noticed at its next write (keep-alive comments come every 10-15 s
 * during a prefill).
 *
 * <p>On Tomcat (the container of the console) the check asks the connection for available input without blocking
 * ({@code ActionCode.AVAILABLE} with a read, what a non-blocking {@code ServletInputStream.isReady} does): once the
 * request body has been read, the socket has nothing more to say until the client sends its next request or closes
 * the connection, and a read that hits the end of the stream (or fails) reports input as available. A byte of a
 * pipelined next request would look the same; no HTTP client pipelines POSTs. Polled at most every {@link #EVERY_NS}
 * from the engine's cancellation checks (between prefill chunks and before every token), on the request's own
 * thread. Elsewhere (another container, HTTP/2, MockMvc) it never fires and the old behaviour stays: streams stop at a
 * failed write.
 */
public final class ClientGone implements BooleanSupplier {

    /** At most one probe per 50 ms: a non-blocking read is a system call. */
    static final long EVERY_NS = 50_000_000L;

    private static final BooleanSupplier NEVER = () -> false;

    private final org.apache.coyote.Request coyote;
    private final AtomicBoolean gone = new AtomicBoolean();
    private long last;

    private ClientGone(org.apache.coyote.Request coyote) {
        this.coyote = coyote;
    }

    /** The check for {@code req}; one that is always false when the container offers none. */
    public static BooleanSupplier of(HttpServletRequest req) {
        if (req.getProtocol() == null || !req.getProtocol().startsWith("HTTP/1")) {
            return NEVER;
        }
        try {
            org.apache.coyote.Request c = Tomcat.coyote(req);
            return c == null ? NEVER : new ClientGone(c);
        } catch (LinkageError | RuntimeException e) { // not Tomcat, or a Tomcat without these internals
            return NEVER;
        }
    }

    @Override
    public boolean getAsBoolean() {
        if (gone.get()) {
            return true;
        }
        long now = System.nanoTime();
        if (last != 0 && now - last < EVERY_NS) {
            return false;
        }
        last = now;
        try {
            if (!coyote.isFinished()) {
                return false; // the body is still being read: available input is the body
            }
            coyote.action(org.apache.coyote.ActionCode.AVAILABLE, Boolean.TRUE);
            if (coyote.getAvailable() > 0) {
                gone.set(true);
            }
        } catch (RuntimeException e) {
            gone.set(true); // the connection is in no state to answer
        }
        return gone.get();
    }

    /** Tomcat specifics, in a class of their own so that nothing of Tomcat is loaded under another container. */
    private static final class Tomcat {
        private static final Field FACADE_REQUEST;

        static {
            Field f;
            try {
                f = org.apache.catalina.connector.RequestFacade.class.getDeclaredField("request");
                f.setAccessible(true);
            } catch (ReflectiveOperationException | RuntimeException e) {
                f = null;
            }
            FACADE_REQUEST = f;
        }

        static org.apache.coyote.Request coyote(ServletRequest req) throws LinkageError {
            while (req instanceof ServletRequestWrapper w) {
                req = w.getRequest();
            }
            if (FACADE_REQUEST == null || !(req instanceof org.apache.catalina.connector.RequestFacade facade)) {
                return null;
            }
            try {
                Object r = FACADE_REQUEST.get(facade);
                return r instanceof org.apache.catalina.connector.Request cr ? cr.getCoyoteRequest() : null;
            } catch (IllegalAccessException e) {
                return null;
            }
        }
    }
}
