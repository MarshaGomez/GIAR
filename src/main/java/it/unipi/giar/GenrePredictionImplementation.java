package it.unipi.giar;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class GenrePredictionImplementation {
	
	String [] names = new String [3];//vector of names
	static Instances [] vettTrain = new Instances [12];// vector of binary datasets
	
	
	private static List<String> predictedGenres = new ArrayList<>();
	
	public static List<String> predictGenres(String descrizione) {
		List<String> genres = new ArrayList<>();

	
		
		return predictedGenres;
	}
	

	
	
	
	
	
	
	
	
	
	
	
}
