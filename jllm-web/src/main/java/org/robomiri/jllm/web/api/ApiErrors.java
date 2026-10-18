package org.robomiri.jllm.web.api;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.io.IOException;
import java.util.Map;
import java.util.NoSuchElementException;

/** Maps the management API's exceptions to {@code {"error": "..."}} responses. */
@RestControllerAdvice(assignableTypes = ManagementController.class)
public class ApiErrors {

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, String>> badRequest(IllegalArgumentException e) {
        return body(HttpStatus.BAD_REQUEST, e);
    }

    @ExceptionHandler(NoSuchElementException.class)
    public ResponseEntity<Map<String, String>> notFound(NoSuchElementException e) {
        return body(HttpStatus.NOT_FOUND, e);
    }

    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<Map<String, String>> conflict(IllegalStateException e) {
        return body(HttpStatus.CONFLICT, e);
    }

    @ExceptionHandler(IOException.class)
    public ResponseEntity<Map<String, String>> io(IOException e) {
        return body(HttpStatus.BAD_GATEWAY, e); // mostly the Hub: unreachable, 401/404 for a repository
    }

    private static ResponseEntity<Map<String, String>> body(HttpStatus status, Exception e) {
        return ResponseEntity.status(status).body(Map.of("error", e.getMessage() == null ? e.toString() : e.getMessage()));
    }
}
