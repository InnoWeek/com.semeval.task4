package com.semeval.task4;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class RnnNetworkTrainer extends AbstractNetworkTrainer {
    private static final int INDEX_WORD_VECTORS_PATH = 0;
    private static final int INDEX_WORD_VECTORS_SIZE = 1;
    private static final int INDEX_TRAIN_SET_PATH = 2;
    private static final int INDEX_TEST_SET_PATH = 3;
    private static final int INDEX_NETWORK_SAVE_PATH = 4;

    private final WordVectors wordVectors;
    protected final int vectorSize;

    public RnnNetworkTrainer(Path trainSet, Path testSet, WordVectors wordVectors, int vectorSize) {
        super(trainSet, testSet);
        this.wordVectors = wordVectors;
        this.vectorSize = vectorSize;
    }

    @Override
    protected MultiLayerNetwork createNetwork() {
        final int nOut = Math.min(vectorSize, 200);
        final MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .updater(Updater.RMSPROP)
                .regularization(true)
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.15)
                .list()
                .layer(0, new GravesBidirectionalLSTM.Builder()
                        .nIn(vectorSize)
                        .nOut(nOut)
                        .activation("softsign")
                        .build())
                .layer(1, new GravesBidirectionalLSTM.Builder()
                        .nIn(vectorSize)
                        .nOut(nOut)
                        .activation("softsign")
                        .build())
                .layer(2, new GravesBidirectionalLSTM.Builder()
                        .nIn(vectorSize)
                        .nOut(nOut)
                        .activation("softsign")
                        .build())
                .layer(3, new GravesBidirectionalLSTM.Builder()
                        .nIn(vectorSize)
                        .nOut(nOut)
                        .activation("softsign")
                        .build())
                .layer(4, new RnnOutputLayer.Builder()
                        .activation("softmax")
                        .lossFunction(LossFunction.MCXENT)
                        .nIn(nOut)
                        .nOut(2)
                        .build())
                .pretrain(false)
                .backprop(true)
                .build();

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(configuration);
        multiLayerNetwork.init();
        return multiLayerNetwork;
    }

    @Override
    protected DataSetIterator createTrainSetIterator(Path trainSet) throws IOException {
        return new TwitterDataIterator(trainSet, wordVectors, vectorSize, 50);
    }

    @Override
    protected DataSetIterator createTestSetIterator(Path testSet) throws IOException {
        return new TwitterDataIterator(testSet, wordVectors, vectorSize, 50);
    }

    public static void main(String[] args) throws TrainingException, IOException {
        final String argWordVectorsPath = args[INDEX_WORD_VECTORS_PATH];
        final String argVectorSize = args[INDEX_WORD_VECTORS_SIZE];

        System.out.println("Loading word vectors (" + argWordVectorsPath + ") ....");
        System.out.println("Vector size: " + argVectorSize);

        final int vectorSize = Integer.parseInt(argVectorSize);
        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(argWordVectorsPath));

        final Path trainSet = Paths.get(args[INDEX_TRAIN_SET_PATH]);
        final Path testSet = Paths.get(args[INDEX_TEST_SET_PATH]);

        System.out.println("Train set: " + trainSet);
        System.out.println("Creating trainer...");
        System.out.println("Test set: " + testSet);

        final RnnNetworkTrainer trainer = new RnnNetworkTrainer(trainSet, testSet, wordVectors, vectorSize);
        trainer.addPreprocessor(new HashTagPreprocessor());
        trainer.addPreprocessor(new UrlRemovingPreprocessor());
        trainer.addPreprocessor(new ReplaceEmoticonsPreprocessor());
        trainer.addPreprocessor(new PunctuationRemovingPreprocessor());
        trainer.addPreprocessor(new RepeatingCharsPreprocessor());
        trainer.addPreprocessor(new ToLowerCasePreprocesor());

        for (int i = 0; i < 8; i++) {
            System.out.println("\nEpoch: " + i);
            System.out.println("Training...");
            trainer.train(1);
            System.out.println("Evaluating...");
            System.out.println(trainer.evaluate().stats());
        }

        try (OutputStream rawOut = Files.newOutputStream(Paths.get(args[INDEX_NETWORK_SAVE_PATH]))) {
            trainer.saveNetwork(rawOut);
        }
    }
}
