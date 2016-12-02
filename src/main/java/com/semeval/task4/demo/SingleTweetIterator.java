package com.semeval.task4.demo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.semeval.task4.TwitterDataIterator;

public class SingleTweetIterator implements DataSetIterator {
	private static final Logger logger = LoggerFactory.getLogger(TwitterDataIterator.class);
	private static final int NUMBER_OF_LABELS = 2;
	private static final String LABEL_NEGATIVE = "negative";
	private static final String LABEL_POSITIVE = "positive";

	private final String tweet;
	private final WordVectors wordVectors;
	private final int batchSize;
	private final int vectorSize;
	private final TokenizerFactory tokenizerFactory;
	private int cursor = 0;

	public SingleTweetIterator(String tweet, WordVectors wordVectors, int batchSize) throws IOException {
		this.tweet = tweet;
		this.wordVectors = wordVectors;
		this.batchSize = batchSize;
		this.vectorSize = wordVectors.lookupTable().layerSize();
		tokenizerFactory = new DefaultTokenizerFactory();
	}

	@Override
	public DataSet next(int num) {
		String[] arr = new String[] { "positive", tweet };
		final List<String[]> tweets = new ArrayList<>();
		tweets.add(arr);

		final List<List<String>> allTokens = new ArrayList<>(tweets.size());
		int maxLength = 0;
		final List<String> tweetTokens = tokenizeTweet(tweet);
		allTokens.add(tweetTokens);
		maxLength = Math.max(maxLength, tweetTokens.size());

		INDArray features = Nd4j.create(tweets.size(), vectorSize, maxLength);

		// We need one label for each classification result - positive/negative
		INDArray labels = Nd4j.create(tweets.size(), NUMBER_OF_LABELS, maxLength);

		// Because the tweets have different lengths, we need to pad the data.
		// "1" means that the data is available, "0" means it's just padding
		INDArray featuresMask = Nd4j.zeros(tweets.size(), maxLength);
		INDArray labelsMask = Nd4j.zeros(tweets.size(), maxLength);

		// Fill in the data into the vectors
		int[] featureMaskIndexes = new int[2];
		for (int tweetIndex = 0; tweetIndex < tweets.size(); tweetIndex++) {
			final List<String> tokens = allTokens.get(tweetIndex);
			featureMaskIndexes[0] = tweetIndex;

			// Add word vector for each token
			for (int tokenIndex = 0; tokenIndex < tokens.size(); tokenIndex++) {
				final String token = tokens.get(tokenIndex);
				final INDArray vector = wordVectors.getWordVectorMatrix(token);
				features.put(new INDArrayIndex[] { NDArrayIndex.point(tweetIndex), NDArrayIndex.all(),
						NDArrayIndex.point(tokenIndex) }, vector);

				// Mark that [tweet,token] is present - i.e. it is not padding
				featureMaskIndexes[1] = tokenIndex;
				featuresMask.putScalar(featureMaskIndexes, 1.0);
			}

			final int labelIndex = getLabelIndex(tweets, tweetIndex);
			int lastIdx = tokens.size();

			// Mark either positive/negative as 1.0 (i.e set the flag)
			labels.putScalar(new int[] { tweetIndex, labelIndex, lastIdx - 1 }, 1.0);

			// Set the mask to 1.0 for the tweet index - i.e it's not padding
			labelsMask.putScalar(new int[] { tweetIndex, lastIdx - 1 }, 1.0);
		}

		return new DataSet(features, labels, featuresMask, labelsMask);
	}

	private List<String> tokenizeTweet(String tweet) {
		final List<String> tokens = tokenizerFactory.create(tweet).getTokens().stream()
				.map(token -> token.toLowerCase()).collect(Collectors.toList());
		final Iterator<String> tokenIterator = tokens.iterator();

		while (tokenIterator.hasNext()) {
			final String token = tokenIterator.next();
			if (!wordVectors.hasWord(token)) {
				tokenIterator.remove();
			}
		}
		return tokens;
	}

	private int getLabelIndex(List<String[]> tweets, int tweetIndex) {
		final int labelIndex;
		final String label = tweets.get(tweetIndex)[0];
		if (label.equalsIgnoreCase(LABEL_NEGATIVE)) {
			labelIndex = 0;
		} else if (label.equalsIgnoreCase(LABEL_POSITIVE)) {
			labelIndex = 1;
		} else {
			throw new IllegalStateException("Unknown label: " + label);
		}

		return labelIndex;
	}

	@Override
	public int totalExamples() {
		return 1;
	}

	@Override
	public int inputColumns() {
		return vectorSize;
	}

	@Override
	public int totalOutcomes() {
		return 2;
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		return true;
	}

	@Override
	public void reset() {
		cursor = 0;
	}

	@Override
	public int batch() {
		return batchSize;
	}

	@Override
	public int cursor() {
		return cursor;
	}

	@Override
	public int numExamples() {
		return totalExamples();
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		throw new UnsupportedOperationException();
	}

	@Override
	public List<String> getLabels() {
		return Arrays.asList("negative", "positive");
	}

	@Override
	public boolean hasNext() {
		return true;
	}

	@Override
	public DataSet next() {
		return next(batchSize);
	}

}
