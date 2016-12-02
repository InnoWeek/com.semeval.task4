package com.semeval.task4.demo;


import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import com.semeval.task4.AbstractNetworkTrainer;
import com.semeval.task4.HashTagPreprocessor;
import com.semeval.task4.PunctuationRemovingPreprocessor;
import com.semeval.task4.RepeatingCharsPreprocessor;
import com.semeval.task4.ReplaceEmoticonsPreprocessor;
import com.semeval.task4.ToLowerCasePreprocesor;
import com.semeval.task4.TrainingException;
import com.semeval.task4.TwitterDataIterator;
import com.semeval.task4.UrlRemovingPreprocessor;

public final class DemoMain extends AbstractNetworkTrainer {
    private final WordVectors wordVectors;
    protected final int vectorSize;
public static final String WORD_VECTORS_PATH = "C:\\Users\\i323283\\Downloads\\glove.twitter.27B\\glove.twitter.27B.200d.txt";
	
	static final File fileToSaveNetworkModel = new File("myRNN.zip");
    public DemoMain(Path trainSet, Path testSet, WordVectors wordVectors, int vectorSize) {
        super(trainSet, testSet);
        trainingStarted = false;
        this.wordVectors = wordVectors;
        this.vectorSize = vectorSize;
    }

    @Override
    protected MultiLayerNetwork createNetwork() {
      
      try {
			return  ModelSerializer.restoreMultiLayerNetwork(fileToSaveNetworkModel);
		} catch (IOException e) {
		}

      return null;
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
    	System.out.println("Loading word vectors (" + WORD_VECTORS_PATH + ") ....");
		WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH));
		int vectorSize = 200;
		String tweet = "positive	a	bad angry worse";
		
		MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(fileToSaveNetworkModel);
		try{
		    PrintWriter writer = new PrintWriter("tweetPath.txt", "UTF-8");
		    writer.println(tweet);
		    writer.close();
		} catch (IOException e) {
		   // do something
		}
        final Path trainSet = Paths.get("tweetPath.txt");
        final Path testSet = Paths.get("tweetPath.txt");


        final DemoMain trainer = new DemoMain(trainSet, testSet, wordVectors, vectorSize);
        trainer.addPreprocessor(new HashTagPreprocessor());
        trainer.addPreprocessor(new UrlRemovingPreprocessor());
        trainer.addPreprocessor(new ReplaceEmoticonsPreprocessor());
        trainer.addPreprocessor(new PunctuationRemovingPreprocessor());
        trainer.addPreprocessor(new RepeatingCharsPreprocessor());
        trainer.addPreprocessor(new ToLowerCasePreprocesor());

        trainer.trainingStarted = true;
        System.out.println(trainer.evaluate(20).stats());
        

//        try (OutputStream rawOut = Files.newOutputStream(Paths.get("C:\\Users\\i323283\\Desktop\\semeval\\twitter_download\\output\\2016.train.bd.txt_semeval_tweets.txt"))) {
//            trainer.saveNetwork(rawOut);
//        }
    }
}
