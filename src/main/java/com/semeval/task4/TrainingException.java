package com.semeval.task4;

public class TrainingException extends Exception {
    public TrainingException() {
        super();
    }

    public TrainingException(String message) {
        super(message);
    }

    public TrainingException(String message, Throwable cause) {
        super(message, cause);
    }

    public TrainingException(Throwable cause) {
        super(cause);
    }

    protected TrainingException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
