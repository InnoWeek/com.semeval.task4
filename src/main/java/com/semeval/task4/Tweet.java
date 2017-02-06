package com.semeval.task4;

public final class Tweet {
    private final String text;
    private final String topic;
    private Sentiment sentiment;

    public Tweet(String text, String topic, Sentiment sentiment) {
        this.text = text;
        this.topic = topic;
        this.sentiment = sentiment;
    }

    public String getText() {
        return text;
    }

    public String getTopic() {
        return topic;
    }

    public Sentiment getSentiment() {
        return sentiment;
    }
}
