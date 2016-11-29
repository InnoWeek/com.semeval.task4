package com.semeval.task4;

import static org.junit.Assert.*;

import org.junit.Test;


public class ReplaceEmoticonsPreprocessorTest {
  
  TweetPreprocessor replaceEmoticonsPreprocessor = new ReplaceEmoticonsPreprocessor();
  
  @Test
  public void tweetsTest() throws Exception {
    String tweetWithSmile = "Tweeter is here :)";
    String tweetWithASadFace = "Feeling hungry :(";
    
    assertEquals("Tweeter is here smile",  replaceEmoticonsPreprocessor.preProcess(tweetWithSmile));
    assertEquals("Feeling hungry sad",  replaceEmoticonsPreprocessor.preProcess(tweetWithASadFace));
  }

}
