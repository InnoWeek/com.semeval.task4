package com.semeval.task4;

import static org.junit.Assert.*;

import org.junit.Test;


public class RepeatingCharsPreprocessorTest {
  
  TweetPreprocessor repeatingCharsPreprocessor = new RepeatingCharsPreprocessor();
  
  @Test
  public void tweetsTest() throws Exception {
    String tweetWithTwoRepeatingChars = "Tweeter is here";
    String tweetWithALotRepeatingChars= "I'm realyyyy hungryyyyy my maaaan";
    
    assertEquals("Tweeter is here",  repeatingCharsPreprocessor.preProcess(tweetWithTwoRepeatingChars));
    assertEquals("I'm realy hungry my man",  repeatingCharsPreprocessor.preProcess(tweetWithALotRepeatingChars));
  }

}
