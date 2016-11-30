package com.semeval.task4;


import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class HashTagPreprocessor implements TweetPreprocessor {
    private static final String REGEX_HASH_TAG = "#(\\S+)";
    private static final String REGEX_CAMEL_CASE = "(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])";
    private static final int GROUP_HASH_TAG_VALUE = 1;
    private final Pattern hashTagPattern;
    private final Pattern camelCasePattern;

    public HashTagPreprocessor() {
        hashTagPattern = Pattern.compile(REGEX_HASH_TAG);
        camelCasePattern = Pattern.compile(REGEX_CAMEL_CASE);
    }

    @Override
    public String preProcess(String tweet) {
        final StringBuilder fixedTweet = new StringBuilder();
        final Matcher hashTagMatcher = hashTagPattern.matcher(tweet);
        int endOfLastMatch = 0;

        while (hashTagMatcher.find()) {
            final int start = hashTagMatcher.start(GROUP_HASH_TAG_VALUE);
            final int end = hashTagMatcher.end(GROUP_HASH_TAG_VALUE);

            final String text = tweet.substring(endOfLastMatch, start - 1);
            fixedTweet.append(text);

            final String hashTag = tweet.substring(start, end);
            final String[] wordsInHashTag = splitHashTag(hashTag);
            for (int i = 0; i < wordsInHashTag.length; i++) {
                if (i != 0 && i != wordsInHashTag.length){
                    fixedTweet.append(" ");
                }
                fixedTweet.append(wordsInHashTag[i]);
            }

            endOfLastMatch = end;
        }

        fixedTweet.append(tweet.substring(endOfLastMatch, tweet.length()));
        return fixedTweet.toString();
    }


    private String[] splitHashTag(String hashTag) {
        return camelCasePattern.split(hashTag);
    }
}
