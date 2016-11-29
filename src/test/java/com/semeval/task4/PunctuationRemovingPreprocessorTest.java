package com.semeval.task4;

import static org.junit.Assert.*;

import org.junit.Test;


public class PunctuationRemovingPreprocessorTest {
  
  TweetPreprocessor punctuationRemovingPreprocessor = new PunctuationRemovingPreprocessor();
  
  @Test
  public void symbolsTest() throws Exception {
    String tweetContainingSymbols = "Hello, there, how are you";
    String tweetContainingMoreSymbols = "Hello. How are you doing ? Just some text# ! @ 123";
    
    assertEquals("Hello there how are you",  punctuationRemovingPreprocessor.preProcess(tweetContainingSymbols));
    assertEquals("Hello How are you doing  Just some text   123",  punctuationRemovingPreprocessor.preProcess(tweetContainingMoreSymbols));
  }

}
