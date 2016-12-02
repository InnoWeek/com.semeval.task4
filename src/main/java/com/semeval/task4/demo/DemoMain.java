package com.semeval.task4.demo;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.semeval.task4.HashTagPreprocessor;
import com.semeval.task4.PunctuationRemovingPreprocessor;
import com.semeval.task4.RepeatingCharsPreprocessor;
import com.semeval.task4.ReplaceEmoticonsPreprocessor;
import com.semeval.task4.ToLowerCasePreprocesor;
import com.semeval.task4.TrainingException;
import com.semeval.task4.TweetPreprocessor;
import com.semeval.task4.UrlRemovingPreprocessor;

public final class DemoMain {
	public static final String WORD_VECTORS_PATH = "C:\\Users\\i323283\\Downloads\\glove.twitter.27B\\glove.twitter.27B.200d.txt";
	static final File fileToSaveNetworkModel = new File("trainedNetwork.zip");

	public static void main(String[] args) throws TrainingException, IOException {
		System.out.println("Loading word vectors (" + WORD_VECTORS_PATH + ") ....");
		WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH));
		System.out.println("Loading network model (" + fileToSaveNetworkModel + ") ....");
		MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(fileToSaveNetworkModel);
		List<TweetPreprocessor> preprocessors = new ArrayList<>();
		preprocessors.add(new HashTagPreprocessor());
		preprocessors.add(new UrlRemovingPreprocessor());
		preprocessors.add(new ReplaceEmoticonsPreprocessor());
		preprocessors.add(new PunctuationRemovingPreprocessor());
		preprocessors.add(new RepeatingCharsPreprocessor());
		preprocessors.add(new ToLowerCasePreprocesor());

		while (true) {

			System.out.print("Enter a tweet: ");
			Scanner scanner = new Scanner(System.in);
			String tweet = scanner.nextLine();

			for (TweetPreprocessor preprocessor : preprocessors) {
				tweet = preprocessor.preProcess(tweet);

			}
			DataSetIterator testIterator = new AsyncDataSetIterator(new SingleTweetIterator(tweet, wordVectors, 100),
					1);

			Evaluation evaluation = new Evaluation();

			DataSet t = testIterator.next();
			INDArray features = t.getFeatureMatrix();
			INDArray lables = t.getLabels();
			INDArray inMask = t.getFeaturesMaskArray();
			INDArray outMask = t.getLabelsMaskArray();
			INDArray predicted = net.output(features, false, inMask, outMask);

			evaluation.evalTimeSeries(lables, predicted, outMask);

			testIterator.reset();

			System.out.print("Evaluation: ");
			if (evaluation.getTopNCorrectCount() == evaluation.getTopNTotalCount()) {
				System.out.println("POSITIVE");
			} else {
				System.out.println("NEGATIVE");
			}
			System.out.println("=================================");
		}

	}

}