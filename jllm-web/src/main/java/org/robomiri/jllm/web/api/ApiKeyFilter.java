package org.robomiri.jllm.web.api;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.robomiri.jllm.web.WebProperties;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/**
 * With {@code jllm.web.api-key} set, {@code /v1/**} and {@code /api/**} need {@code Authorization: Bearer <key>}
 * (the console asks for the key once and keeps it in the browser). The static UI itself stays public.
 */
@Component
public class ApiKeyFilter extends OncePerRequestFilter {

    private final byte[] key;

    public ApiKeyFilter(WebProperties props) {
        this.key = props.apiKeyRequired() ? props.apiKey().getBytes(StandardCharsets.UTF_8) : null;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String p = request.getRequestURI().substring(request.getContextPath().length());
        // generated pictures are fetched by <img> tags, which cannot send the key: their unguessable URL is the capability
        return key == null || "OPTIONS".equals(request.getMethod()) || !(p.startsWith("/v1/") || p.startsWith("/api/"))
                || (p.startsWith("/v1/images/files/") && "GET".equals(request.getMethod()));
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {
        String auth = request.getHeader("Authorization");
        String given = auth != null && auth.regionMatches(true, 0, "Bearer ", 0, 7) ? auth.substring(7).trim() : "";
        if (!MessageDigest.isEqual(key, given.getBytes(StandardCharsets.UTF_8))) {
            response.setStatus(401);
            response.setHeader("WWW-Authenticate", "Bearer");
            response.setContentType("application/json");
            response.getWriter().write("{\"error\":{\"message\":\"missing or invalid API key\",\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key\"}}");
            return;
        }
        chain.doFilter(request, response);
    }
}
