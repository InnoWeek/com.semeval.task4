package com.semeval.task4;

public final class RepeatingCharsPreprocessor implements TweetPreprocessor {

    private static final String REGEX_REPEATING_SYMBOLS_AT_LEAST_3_TIMES = "([a-zA-Z])(\\1{3,})";

    @Override
    public String preProcess(String tweet) {
        return tweet.replaceAll(REGEX_REPEATING_SYMBOLS_AT_LEAST_3_TIMES, "$1");
    }
}
