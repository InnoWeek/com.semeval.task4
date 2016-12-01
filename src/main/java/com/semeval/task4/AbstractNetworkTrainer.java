package com.semeval.task4;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

public abstract class AbstractNetworkTrainer {
    private static final String CLASSIFICATION_POSITIVE = "positive";
    private static final String CLASSIFICATION_NEGATIVE = "negative";
    private static final String POSITIVE_VALUE = "1";
    private static final String NEGATIVE_VALUE = "-1";

    private final List<TweetPreprocessor> preprocessors;
    private final Path trainSet;
    private final Path testSet;

    private MultiLayerNetwork network;
    private boolean trainingStarted;
    private DataSetIterator trainSetIterator;
    private DataSetIterator testSetIterator;

    public AbstractNetworkTrainer(Path trainSet, Path testSet) {
        preprocessors = new ArrayList<>();
        this.trainSet = trainSet;
        this.testSet = testSet;

        if (!Files.isRegularFile(trainSet)) {
            throw new IllegalArgumentException("The train set must be a file: " + testSet);
        }

        if (!Files.isReadable(testSet)) {
            throw new IllegalArgumentException("The test set must be a file: " + testSet);
        }
    }

    protected abstract MultiLayerNetwork createNetwork();

    protected abstract DataSetIterator createTrainSetIterator(Path trainSet) throws IOException;

    protected abstract DataSetIterator createTestSetIterator(Path testSet) throws IOException;


    public final void addPreprocessor(TweetPreprocessor preprocessor) {
        if (trainingStarted) {
            throw new IllegalStateException("Cannot add more preprocessors after training has started.");
        }
        preprocessors.add(preprocessor);
    }

    private Path preProcessDataSet(Path dataSet) throws IOException {
        final Path preProcessedDataSet = Files.createTempFile("networkTrainer", ".txt");
        try (Stream<String> lines = Files.lines(dataSet)) {
            /**
             * Split the data columns to an array
             */
            Stream<String[]> preProcessingPipeline = lines.map(l -> l.split("\t+"))
                    /**
                     * Some of the tweets are missing and "not available"
                     * is set instead of teh tweet text. They must be removed
                     */
                    .filter(l -> !l[3].equalsIgnoreCase("not available"))

                    /**
                     * Make sure the topic is in lower case
                     */
                    .peek(l -> l[1] = l[1].toLowerCase(Locale.ENGLISH))

                    /**
                     * Replace the long text based classifications with
                     * shorter number based ones in order to save memory
                     */
                    .peek(l -> l[2] = translateClassification(l[2]));

            /**
             * Run all configured preprocessors against the tweets
             */
            for (TweetPreprocessor preprocessor : preprocessors) {
                preProcessingPipeline = preProcessingPipeline.peek(l -> {
                    l[3] = preprocessor.preProcess(l[3]);
                });
            }

            /**
             * Write out the preprocessed data
             */
            try (BufferedWriter out = Files.newBufferedWriter(preProcessedDataSet, StandardCharsets.UTF_8)) {
                preProcessingPipeline.forEach(l -> {
                    writePreprocessedData(out, l);
                });
            }

            return preProcessedDataSet;
        } catch (UncheckedIOException ex) {
            throw new IOException(ex.getCause());
        }
    }

    private String translateClassification(String classification) {
        if (classification.equalsIgnoreCase(CLASSIFICATION_POSITIVE)) {
            return POSITIVE_VALUE;
        } else if (classification.equalsIgnoreCase(CLASSIFICATION_NEGATIVE)) {
            return NEGATIVE_VALUE;
        } else {
            throw new IllegalStateException("Encountered unknown classification: " + classification);
        }
    }

    private void writePreprocessedData(BufferedWriter out, String[] row) {
        try {
            out.write(row[2]);
            out.write("\t");
            out.write(row[1]);
            out.write("\t");
            out.write(row[3]);
            out.write("\n");
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }


    public final void train(int epochs, int batchSize) throws TrainingException {
        try {
            if (!trainingStarted) {
                final Path preProcessedTrainSet = preProcessDataSet(trainSet);
                trainSetIterator = createTrainSetIterator(preProcessedTrainSet);
                network = createNetwork();
                trainingStarted = true;
            }
            trainingStarted = true;

            for (int epoch = 0; epoch < epochs; epoch++) {
                network.fit(trainSetIterator);
                trainSetIterator.reset();
            }
        } catch (IOException ex) {
            throw new TrainingException(ex);
        }
    }

    public final Evaluation evaluate(int batchSize) throws TrainingException {
        if (!trainingStarted) {
            throw new IllegalStateException("A network must be trained first");
        }

        try {
            if (null == testSetIterator) {
                final Path preProcessedTestSet = preProcessDataSet(testSet);
                testSetIterator = createTestSetIterator(preProcessedTestSet);
            }

            final Evaluation evaluation = network.evaluate(testSetIterator);
            testSetIterator.reset();
            return evaluation;
        } catch (IOException ex) {
            throw new TrainingException(ex);
        }
    }

    public final void saveNetwork(OutputStream destination) throws IOException {
        if (!trainingStarted) {
            throw new IllegalStateException("A network must be trained first");
        }

        ModelSerializer.writeModel(network, destination, true);
    }
}
