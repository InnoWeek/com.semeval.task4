package com.semeval.task4;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class HashTagPreprocessorTest {
    private HashTagPreprocessor preprocessor;

    @BeforeEach
    public void beforeEach() throws Exception {
        preprocessor = new HashTagPreprocessor();
    }

    @Test
    void testPreProcessWithNoHashTags() {
        final String expected = "There are no HashTags here!";
        final String actual = preprocessor.preProcess(expected);

        assertEquals(expected, actual);
    }


    @Test
    void testPreProcessWithHashTagAtTheBeginning() {
        final String input = "#SemevalVictory HashTags at the beginning.";
        final String expected = "Semeval Victory HashTags at the beginning.";

        final String actual = preprocessor.preProcess(input);

        assertEquals(expected, actual);
    }

    @Test
    void testPreProcessWithHashTagAtTheEnd() {
        final String input = "Hey, #WeLoveHashTags";
        final String expected = "Hey, We Love Hash Tags";

        final String actual = preprocessor.preProcess(input);

        assertEquals(expected, actual);
    }

    @Test
    void testPreProcessWithHashTagInTheMiddled() {
        final String input = "Hey, #WeLoveHashTags, because they are good";
        final String expected = "Hey, We Love Hash Tags, because they are good";

        final String actual = preprocessor.preProcess(input);

        assertEquals(expected, actual);
    }

    @Test
    void testPreProcessWithSeveralHashTags() {
        final String input = "Hey, #WeLoveHashTags, because #theyAreGood";
        final String expected = "Hey, We Love Hash Tags, because they Are Good";

        final String actual = preprocessor.preProcess(input);

        assertEquals(expected, actual);
    }

}
