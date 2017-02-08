package com.semeval.task4;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 *
 */
public final class Demo {
    private static final int INDEX_WORD_VECTORS_PATH = 0;
    private static final int INDEX_WORD_VECTORS_SIZE = 1;
    private static final int INDEX_NETWORK_PATH = 2;

    public static void main(String[] args) throws IOException {
        final String argWordVectorsPath = args[INDEX_WORD_VECTORS_PATH];
        final String argVectorSize = args[INDEX_WORD_VECTORS_SIZE];

        System.out.println("Loading word vectors (" + argWordVectorsPath + ") ....");
        System.out.println("Vector size: " + argVectorSize);

        final int vectorSize = Integer.parseInt(argVectorSize);
        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(argWordVectorsPath));

        final String networkPath = args[INDEX_NETWORK_PATH];
        final MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(networkPath);

        final List<TweetPreprocessor> preprocessors = new ArrayList<>();
        preprocessors.add(new HashTagPreprocessor());
        preprocessors.add(new UrlRemovingPreprocessor());
        preprocessors.add(new ReplaceEmoticonsPreprocessor());
        preprocessors.add(new PunctuationRemovingPreprocessor());
        preprocessors.add(new RepeatingCharsPreprocessor());
        preprocessors.add(new ToLowerCasePreprocesor());

        final Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("Topic: ");
            final String topic = scanner.nextLine();
            if(topic.isEmpty()){
                System.out.println("You must enter topic");
                continue;
            }

            System.out.print("Text: ");
            String text = scanner.nextLine();

            for (TweetPreprocessor preprocessor : preprocessors) {
                text = preprocessor.preProcess(text);
            }

            if(text.isEmpty()){
                System.out.println("You must enter text");
                continue;
            }

            final Path dataPath = Files.createTempFile("demo", "");
            try (BufferedWriter bufferedWriter = Files.newBufferedWriter(dataPath, StandardCharsets.UTF_8)) {
                //id (empty)
                bufferedWriter.write("\t");

                //topic
                bufferedWriter.write(topic.trim().toLowerCase());
                bufferedWriter.write("\t");

                // sentiment(empty)
                bufferedWriter.write("positive");
                bufferedWriter.write("\t");

                // content
                bufferedWriter.write(text);
                bufferedWriter.write("\n");
            }

            final TwitterDataIterator iterator = new TwitterDataIterator(dataPath, wordVectors, vectorSize, 1);
            final Evaluation evaluation = net.evaluate(iterator);
            System.out.println("Evaluation: " + (evaluation.getTopNCorrectCount() == 1 ? "POSITIVE" : "NEGATIVE"));
            net.rnnClearPreviousState();
        }
    }
}

/**
 * topics: france, england
 * case 1: X is awesome/terrible
 * case 3: X is awesome but Y is terrible
 */
