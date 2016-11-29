
package com.semeval.task4;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public final class ReplaceEmoticonsPreprocessor implements TweetPreprocessor {

  private static Map<String, String> emoticons = new HashMap<String, String>();

  public ReplaceEmoticonsPreprocessor() {
    initializeMap();
  }

  @Override
  public String preProcess(String tweet) {
    String result = tweet;
    for (Entry<String, String> emoticon : emoticons.entrySet()) {
      result = result.replaceAll(emoticon.getKey(), emoticon.getValue());
    }
    return result;

  }

  private static void initializeMap() {
    emoticons.put(":\\)", "smile");
    emoticons.put(":-\\)", "smile");
    emoticons.put("8\\)", "smile");
    emoticons.put("8-\\)", "smile");

    emoticons.put(":\\(", "sad");
    emoticons.put(";\\(", "sad");
    emoticons.put(":-\\(", "sad");
    emoticons.put(";-\\(", "sad");
    emoticons.put(":-<", "sad");

    emoticons.put(":P", "fun");
    emoticons.put(":p", "fun");

    emoticons.put(":-\\*", "kiss");
    emoticons.put(":\\*", "kiss");

    emoticons.put(":-O", "supprise");

    emoticons.put(":d", "laugh");
    emoticons.put(":-d", "laugh");
    emoticons.put(":D", "laugh");
    emoticons.put(":-D", "laugh");

    emoticons.put("O\\.o", "confused");

    emoticons.put(":X", "secret");
    emoticons.put(":-X", "secret");

    emoticons.put(":-/", "sarcasm");

  }
}
