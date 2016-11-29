package com.semeval.task4;

public final class PunctuationRemovingPreprocessor implements TweetPreprocessor {

    private static final String REGEX_PUNTUATION_SYMBOLS = "\\p{Punct}";

    @Override
    public String preProcess(String tweet) {
        return tweet.replaceAll(REGEX_PUNTUATION_SYMBOLS, "");
    }
}
