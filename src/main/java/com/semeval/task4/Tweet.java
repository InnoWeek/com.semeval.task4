package com.semeval.task4;

public final class Tweet {
    private final String text;
    private Sentiment sentiment;

    public Tweet(String text, Sentiment sentiment) {
        this.text = text;
        this.sentiment = sentiment;
    }

    public String getText() {
        return text;
    }

    public Sentiment getSentiment() {
        return sentiment;
    }
}
