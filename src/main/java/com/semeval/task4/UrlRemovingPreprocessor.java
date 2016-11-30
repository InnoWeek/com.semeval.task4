package com.semeval.task4;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class UrlRemovingPreprocessor implements TweetPreprocessor {
    private static final String REGEX_URL = "\\b(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]";

    private final Pattern urlPattern;

    public UrlRemovingPreprocessor() {
        urlPattern = Pattern.compile(REGEX_URL);
    }

    @Override
    public String preProcess(String tweet) {
        final Matcher matcher = urlPattern.matcher(tweet);
        return matcher.replaceAll("");
    }
}
