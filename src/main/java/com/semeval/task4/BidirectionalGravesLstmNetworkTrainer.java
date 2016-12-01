package com.semeval.task4;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
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

public final class BidirectionalGravesLstmNetworkTrainer extends AbstractNetworkTrainer {
    private final WordVectors wordVectors;
    protected final int vectorSize;

    public BidirectionalGravesLstmNetworkTrainer(Path trainSet, Path testSet, WordVectors wordVectors, int vectorSize) {
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
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .learningRate(0.15)
                .list()
                .layer(0, new GravesBidirectionalLSTM.Builder()
                        .nIn(vectorSize)
                        .nOut(nOut)
                        .activation("softsign")
                        .build())
                .layer(1, new RnnOutputLayer.Builder()
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
        System.out.println("Loading word vectors (" + args[0] + ") ....");
        System.out.println("Vector size: " + args[1]);
        final int vectorSize = Integer.parseInt(args[1]);
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(args[0]));

        final Path trainSet = Paths.get(args[2]);
        final Path testSet = Paths.get(args[3]);

        System.out.println("Train set: " + trainSet);
        System.out.println("Creating trainer...");
        System.out.println("Test set: " + testSet);

        final BidirectionalGravesLstmNetworkTrainer trainer = new BidirectionalGravesLstmNetworkTrainer(trainSet, testSet, wordVectors, vectorSize);
        trainer.addPreprocessor(new HashTagPreprocessor());
        trainer.addPreprocessor(new UrlRemovingPreprocessor());
        trainer.addPreprocessor(new ReplaceEmoticonsPreprocessor());
        trainer.addPreprocessor(new PunctuationRemovingPreprocessor());
        trainer.addPreprocessor(new RepeatingCharsPreprocessor());
        trainer.addPreprocessor(new ToLowerCasePreprocesor());

        for (int i = 0; i < 12; i++) {
            System.out.println("\nEpoch: " + i);
            System.out.println("Training...");
            trainer.train(1, 20);
            System.out.println("Evaluating...");
            System.out.println(trainer.evaluate(20).stats());
        }

        try (OutputStream rawOut = Files.newOutputStream(Paths.get("C:\\Users\\i304680\\git\\network.out"))) {
            trainer.saveNetwork(rawOut);
        }
    }
}
