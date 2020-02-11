package it.unipi.giar;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;
public class GenrePredictionImplementation {

	static List<String> genres = new ArrayList<>();
	static{ 
		genres.add("Puzzle");
		genres.add("Adventure");
		genres.add("Action");
		genres.add("RPG");
		genres.add("Simulation");
		genres.add("Strategy");
		genres.add("Shooter");
		genres.add("Sport");
		genres.add("Racing");
		genres.add("Educational");
		genres.add("Fighting");
		genres.add("BoardGames");
	}

	public static List<String> predictGenres(String descrizione) {
		List<String> predictedGenres = new ArrayList<String>();
		try {	
			//pulizia della descrizione 	
			descrizione = descrizione.replaceAll("(https?http|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]", " ");
			descrizione = descrizione.replaceAll("\\s+", " ");

			//crea dataset con unica istanza description
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			ArrayList<String> labels = new ArrayList<String>();
			labels.add("other");

			attributes.add(new Attribute("description", true));
			attributes.add(new Attribute("genre", labels));

			Instances unlabeled = new Instances("Try", attributes, 1);	//1 num istanze
			unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
			// adding instances			
			double[] val = new double[2];
			val[0] = unlabeled.attribute(0).addStringValue(descrizione);
			val[1] = 0;
			
			unlabeled.add(new DenseInstance(1.0, val));
			unlabeled.instance(0).setClassMissing(); 

			//per ogni modello appartenente alla cartella  VA FATTO COI MODELLI CON CONFUSION MATRIX MIGLIORE ETCECT
			for(int z = 0; z < genres.size(); z++) {
				NaiveBayesMultinomialText NBMT;

				NBMT = (NaiveBayesMultinomialText)SerializationHelper.read("./src/main/resources/models/"+ genres.get(z) + ".model");
				System.out.println();
				if(NBMT.classifyInstance(unlabeled.instance(0)) == 0) {
					String predgen = genres.get(z);
					predictedGenres.add(predgen);	        
				}	
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return predictedGenres;
	}
}
