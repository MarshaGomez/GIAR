package it.unipi.giar;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;


public class GenrePrediction {
	
	//String [] names = new String [3];//vector of names
	static Instances [] vettTrain = new Instances [12];// vector of binary datasets
	
	private static List<String> predictedGenres = new ArrayList<>();
	
	public static List<String> init(String descrizione) {
		List<String> genres = new ArrayList<>();
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
		
		
		createDatasets(genres);
		createModels(genres);

		
		predictedGenres.add("prova");
		return predictedGenres;
	}
	
	public static void createDatasets( List<String> genres){

			// Reading Entire Dataset
			DataSource source;
			try {
				source = new DataSource("src/main/resources/dataset.arff");
				Instances data = source.getDataSet();
			
			//splittin 80% 20%
			data.randomize(new java.util.Random());	// randomize instance order before splitting dataset
			Instances train = data.trainCV(5, 0);//5 folds, 100/5=20
			Instances test = data.testCV(5, 0);
			
			//Generating training and test set  	 
			// Make the last attribute be the class
			train.setClassIndex(train.numAttributes() - 1);
			test.setClassIndex(test.numAttributes() - 1);
			
			System.out.println("train size");
			System.out.println(train.size());
			System.out.println("test size");
			System.out.println(test.size());	

			//Collecting the parameters
			int numAttributes = 2;
			int numInstances = train.size();		
			
			//CREATION OF 12 BINARY DATATSETS 
			createBinaryDatasets(genres, numAttributes, numInstances, train);				
			
			
			
			
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	}
	
	
	public static void createBinaryDatasets( List<String> genres, int numAttributes, int numInstances, Instances train) {
		
		for(int z = 0; z < genres.size(); z++) {	//12 generi			
			String genre = genres.get(z);	//genere per il quale sto creando il dataset binario
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			ArrayList<String> labels = new ArrayList<String>();
			labels.add(genres.get(z));
			labels.add("other");
			
			attributes.add(new Attribute("description", true));
			attributes.add(new Attribute("genre", labels));
			
			Instances binTrainDataset = new Instances("Try", attributes, 200);
			binTrainDataset.setClassIndex(binTrainDataset.numAttributes() - 1);
			// adding instances
			String[] values = null;				
			int class_count = 0;
			
			//insert the rows with genre!=other
			for ( int j = 0; j < numInstances; j++ ){	//per ogni riga del db				
				double[] val = new double[2];
				val[0] = binTrainDataset.attribute(0).addStringValue(train.instance(j).stringValue(0));	//val0 prende la descr
				
				if(train.instance(j).stringValue(train.numAttributes() - 1).equals(genres.get(z))) {
					val[1] = 0; 	//val1 prende 
					binTrainDataset.add(new DenseInstance(1.0, val));
					class_count++;
				}
//				System.out.println(binTrainDataset.toString());				
			}
			//insert the rows with genre=other
			for ( int j = 0; j < numInstances; j++ ){	//per ogni riga del db				
				double[] val = new double[2];
				val[0] = binTrainDataset.attribute(0).addStringValue(train.instance(j).stringValue(0));
				
				if(!train.instance(j).stringValue(train.numAttributes() - 1).equals(genres.get(z))) {
					if(class_count > 0) {
						val[1] = 1;
						binTrainDataset.add(new DenseInstance(1.0, val));
						class_count--;
					}
				}
				
				if(class_count == 0) {
					break;
				}			
			}
			
			/*
			//save in arff files to check on weka the results
			ArffSaver saver = new ArffSaver();
			saver.setInstances(binTrainDataset);
			try {
				saver.setFile(new File("src/main/resources/" + genres.get(z) + "_dataset.arff"));
				saver.writeBatch();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			*/
			vettTrain[z] = binTrainDataset;	//save the db in the dbarray at genre z position	
		}
	}
	
	
	public static void createModels(List<String> genres) {	
		//DOPO VA MESSO IN UN FOR CON LE OPZIONI(?)
		trainNaiveBayesModel(genres);
		//trainSMOModel(genres);
		//trainRandomForestModel(genres);
	}
	
	public static void trainNaiveBayesModel(List<String> genres){
		try {
			for(int z = 0; z < genres.size(); z++) {
				
				// Reading Entire Dataset
				DataSource source;
				source = new DataSource("src/main/resources/"+ genres.get(z) + "_dataset.arff");
				Instances naiveTrain = source.getDataSet();
				
				
				naiveTrain.setClassIndex(naiveTrain.numAttributes() - 1);
				 // Building the classifier
				 String[] options = weka.core.Utils.splitOptions("-W -P 0 -M 2.0 -norm 1.0 -lnorm 2.0 -lowercase -stopwords-handler weka.core.stopwords.Rainbow -tokenizer weka.core.tokenizers.AlphabeticTokenizer -stemmer \"weka.core.stemmers.SnowballStemmer -S porter\"");
				 NaiveBayesMultinomialText naive = new NaiveBayesMultinomialText();
				 naive.setOptions(options);
				 naive.buildClassifier(naiveTrain);
				 // Evaluation on the training set
				 Evaluation eval = new Evaluation(naiveTrain);
				 eval.evaluateModel(naive,naiveTrain);
				 System.out.println(eval.toSummaryString("Results Training:\n", false));
				 
				 
				 /*
				 // Evaluation on the test set
				 DataSource source2 = new DataSource("/Users/pietroducange/Desktop/irisTest.arff");
				 Instances test = source2.getDataSet();
				 test.setClassIndex(test.numAttributes() - 1);
				 Evaluation evalTs = new Evaluation(naiveTrain);
				 evalTs.evaluateModel(naive,test);
				 System.out.println(evalTs.toSummaryString("Results Test:\n", false));
				 System.out.println(evalTs.toMatrixString());
				 System.out.println(evalTs.pctCorrect());
				*/
				

				
			}
				
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
					
		
		
	}
	
	
}

