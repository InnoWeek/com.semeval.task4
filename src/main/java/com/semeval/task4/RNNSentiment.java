
package com.semeval.task4;

import java.io.File;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
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
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class RNNSentiment {

  /**
   * Number of epochs (full passes of training data) to train on
   */
  private static final int EPOCHS_NUMBER = 2;
  /**
   * Size of the word vectors. 300 in the Google News model
   */

  private static final int VECTOR_SIZE = 200;
  /**
   * Number of examples in each minibatch
   */
  private static final int BATCH_SIZE = 50;

  static final File fileToSaveNetworkModel = new File("C:\\Users\\i319962\\Documents\\Projects\\Inoweek2016 - Tweets\\myRNN.zip");

  /** Data URL for downloading */
  public static final String DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
  /** Location to save and extract the training/testing data */
  public static final String TRAIN_PATH = "C:\\Users\\i319962\\Documents\\Projects\\Inoweek2016 - Tweets\\tweets.txt";
  public static final String TEST_PATH = "C:\\Users\\i319962\\Documents\\Projects\\Inoweek2016 - Tweets\\tweets.txt";

  // public static final String WORD_VECTORS_PATH =
  // "C:\\Users\\i319962\\Downloads\\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin";
  public static final String WORD_VECTORS_PATH = "C:\\Users\\i319962\\Documents\\Projects\\Inoweek2016 - Tweets\\glove.twitter.27B\\glove.twitter.27B.200d.txt";
  // public static final String WORD_VECTORS_PATH = "C:\\Users\\i319962\\Documents\\Projects\\Inoweek2016 -
  // Tweets\\word2vec_twitter_model.tar\\word2vec_twitter_model\\word2vec_twitter_model.bin";

  public static void main(String[] args) throws Exception {

    boolean loadExistingNet = false;
    MultiLayerNetwork net;

    // Set up network configuration
    if (loadExistingNet) {
      net = ModelSerializer.restoreMultiLayerNetwork(fileToSaveNetworkModel);
    } else {
      net = initializeNeuralNetwork();
    }

    net.setListeners(new ScoreIterationListener(1));

    WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH));
    // WordVectors wordVectors = WordVectorSerializer.loadGoogleModel(new File(WORD_VECTORS_PATH), true);
    DataSetIterator train = new AsyncDataSetIterator(new TwitterDataIterator(TRAIN_PATH, wordVectors, BATCH_SIZE), 1);
    DataSetIterator test = new AsyncDataSetIterator(new TwitterDataIterator(TEST_PATH, wordVectors, 150), 1);

    System.out.println("Starting training");

    netTrainingAndEvaluation(net, train, test);

    System.out.println("----- Example complete -----");

    ModelSerializer.writeModel(net, fileToSaveNetworkModel, false);
  }

  private static MultiLayerNetwork initializeNeuralNetwork() {
    MultiLayerNetwork net;
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .iterations(1).updater(Updater.RMSPROP)
        // .regularization(true).l2(1e-5)
        .weightInit(WeightInit.XAVIER).gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
        .learningRate(0.0018).list().layer(0, new GravesLSTM.Builder().nIn(VECTOR_SIZE).nOut(200).activation("softsign").build())
        .layer(1, new RnnOutputLayer.Builder().activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(2).build())
        .pretrain(false).backprop(true).build();

    net = new MultiLayerNetwork(conf);
    net.init();
    return net;
  }

  private static void netTrainingAndEvaluation(MultiLayerNetwork net, DataSetIterator train, DataSetIterator test) {
    for (int i = 0; i < EPOCHS_NUMBER; i++) {
      net.fit(train);
      train.reset();
      System.out.println("Epoch " + i + " complete. Starting evaluation:");

      Evaluation evaluation = new Evaluation();
      while (test.hasNext()) {
        DataSet t = test.next();
        INDArray features = t.getFeatureMatrix();
        INDArray lables = t.getLabels();
        INDArray inMask = t.getFeaturesMaskArray();
        INDArray outMask = t.getLabelsMaskArray();
        INDArray predicted = net.output(features, false, inMask, outMask);

        evaluation.evalTimeSeries(lables, predicted, outMask);
      }
      test.reset();

      System.out.println(evaluation.stats());
    }
  }

}
