package com.semeval.task4;

import java.util.Locale;

public final class ToLowerCasePreprocesor implements TweetPreprocessor {
    @Override
    public String preProcess(String tweet) {
        return tweet.toLowerCase(Locale.ENGLISH);
    }
}
