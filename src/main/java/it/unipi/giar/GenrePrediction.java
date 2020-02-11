package it.unipi.giar;

import java.io.File;
import java.io.FileNotFoundException;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class GenrePrediction {
	
	static Instances [][] vettTrain = new Instances [10][12];	// vector of binary datasets TRAIN ONE PER EACH FOLD
	static Instances [] vettTest = new Instances [10];	// vector of datasets TEST ONE PER EACH FOLD
	
	static NaiveBayesMultinomialText [] vettNaive = new NaiveBayesMultinomialText[12];
	static FilteredClassifier [] vettSMO = new FilteredClassifier[12];
	static FilteredClassifier [] vettRandomForest = new FilteredClassifier[12];
	
	static int[][] confusionMatrix = new int[12][12];
	static int[][] confusionMatrixSMO = new int[12][12];
	static int[][] confusionMatrixRF = new int[12][12];
	
	private static List<String> predictedGenres = new ArrayList<>();
	static{
		predictedGenres.add("prova");
	}
	
	public static void resetMatrix(int[][] mat) {
		for(int i=0; i<12; i++) {
			for(int j=0; j<12; j++) {
				mat[i][j]=0;
			}
		}
	}
	
	public static void printmatrix (int[][] mat, String classifier) {
		System.out.println("CONFUSION MATRIX " + classifier +"\n");
	
		for(int i=0; i<12; i++) {
			for(int j=0; j<12; j++) {
				System.out.print(mat[i][j] + " ");
			}
			System.out.print("\n");
		}
		System.out.print("\n");
	}
	
	public static List<String> init(String descrizione) {
		List<String> genres = new ArrayList<>();
		genres.add("Puzzle");
		genres.add("Adventure");
		genres.add("Action");
		genres.add("RPG");
		genres.add("Simulation");
		genres.add("Strategy");
		genres.add("Shooter");
		genres.add("Sports");
		genres.add("Racing");
		genres.add("Educational");
		genres.add("Fighting");
		genres.add("BoardGames");
		
		
		//inizializza array di classificatori
		for(int z = 0; z < genres.size(); z++) {		
			String[] options;
			try {
				//naive
				options = weka.core.Utils.splitOptions("-W -P 0 -M 2.0 -norm 1.0 -lnorm 2.0 -lowercase -stopwords-handler weka.core.stopwords.Rainbow -tokenizer weka.core.tokenizers.AlphabeticTokenizer -stemmer \"weka.core.stemmers.SnowballStemmer -S porter\"");		
				vettNaive[z]= new NaiveBayesMultinomialText();
				vettNaive[z].setOptions(options);
				
				//smo
				options = weka.core.Utils.splitOptions("-F \"weka.filters.MultiFilter -F \\\"weka.filters.unsupervised.attribute.StringToWordVector -R 1 -W 1000 -prune-rate -1.0 -I -N 0 -L -stemmer \\\\\\\"weka.core.stemmers.SnowballStemmer -S porter\\\\\\\" -stopwords-handler weka.core.stopwords.Rainbow -M 2 -tokenizer weka.core.tokenizers.AlphabeticTokenizer\\\" -F \\\"weka.filters.supervised.attribute.AttributeSelection -E \\\\\\\"weka.attributeSelection.InfoGainAttributeEval \\\\\\\" -S \\\\\\\"weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1\\\\\\\"\\\"\" -S 1 -W weka.classifiers.functions.SMO -- -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\"");
				vettSMO[z]= new FilteredClassifier();
				vettSMO[z].setOptions(options);
				
				//randomForest				
				options = weka.core.Utils.splitOptions("-F \"weka.filters.MultiFilter -F \\\"weka.filters.unsupervised.attribute.StringToWordVector -R 1 -W 1000 -prune-rate -1.0 -I -N 0 -L -stemmer \\\\\\\"weka.core.stemmers.SnowballStemmer -S porter\\\\\\\" -stopwords-handler weka.core.stopwords.Rainbow -M 2 -tokenizer weka.core.tokenizers.AlphabeticTokenizer\\\" -F \\\"weka.filters.supervised.attribute.AttributeSelection -E \\\\\\\"weka.attributeSelection.InfoGainAttributeEval \\\\\\\" -S \\\\\\\"weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1\\\\\\\"\\\"\" -S 1 -W weka.classifiers.trees.RandomForest -- -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 944723357");
				vettRandomForest[z]= new FilteredClassifier();
				vettRandomForest[z].setOptions(options);
				
				
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		}
		
		resetMatrix(confusionMatrix);
		resetMatrix(confusionMatrixSMO);
		resetMatrix(confusionMatrixRF);
		
		createDatasets(genres);

		
		printmatrix(confusionMatrix,"naive");
		printmatrix(confusionMatrixSMO,"smo");
		printmatrix(confusionMatrixRF,"randomforest");
		
		//esporta tutti  modelli che sono nellarrray dei classificatori naive
		for(int z = 0; z < genres.size(); z++) {		
			try {
				SerializationHelper.write(new FileOutputStream("./src/main/resources/models/"+genres.get(z)+".model"), vettNaive[z]);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		}
		
		return predictedGenres;
	}
	
	public static void createDatasets( List<String> genres){

			// Reading Entire Dataset
			DataSource source;
			try {
				source = new DataSource("src/main/resources/dataset400.arff");
				Instances data = source.getDataSet();			
				data.setClassIndex(data.numAttributes()-1);
				
				// Randomize and stratify the dataset
				data.randomize(new Random(1)); 	 // randomize instance order before splitting dataset
				data.stratify(10);	// 10 folds
				
				for(int i=0; i<10; i++){ // To calculate the results in each fold
					
					Instances test = data.testCV(10, i);
					Instances train = data.trainCV(10, i);
 
					// Make the last attribute be the class
					train.setClassIndex(train.numAttributes() - 1);
					test.setClassIndex(test.numAttributes() - 1);
					
					//TO CHECK
					vettTest[i]=test;
					
					//System.out.println(i + " train size" + train.size());	
					//System.out.println(i + " test size" + test.size());	
		
					int numInstancesTrain = train.size();
				
					//CREATION OF 12 BINARY DATATSETS (repeats this for every fold)
					createBinaryDatasets( genres, numInstancesTrain, train, i, test);				
				}	
					
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	}
	

	public static void createBinaryDatasets( List<String> genres,  int numInstances, Instances train, int foldnum, Instances test) {
		try {
			for(int z = 0; z < genres.size(); z++) {	//12 generi			
				ArrayList<Attribute> attributes = new ArrayList<Attribute>();
				ArrayList<String> labels = new ArrayList<String>();
				labels.add(genres.get(z));
				labels.add("other");

				attributes.add(new Attribute("description", true));
				attributes.add(new Attribute("genre", labels));

				Instances binTrainDataset = new Instances("Try", attributes, 8000);
				binTrainDataset.setClassIndex(binTrainDataset.numAttributes() - 1);

				// adding instances		
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
						saver.setFile(new File("src/main/resources/folds/fold" +foldnum +"/" + genres.get(z) + "_dataset.arff"));
						saver.writeBatch();
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				 */
				

				vettTrain[foldnum][z] = binTrainDataset;	//save the db in the dbarray at genre z position	

				// retrain the classifier for this binary
				vettNaive[z].buildClassifier(binTrainDataset);
				vettSMO[z].buildClassifier(binTrainDataset);
				vettRandomForest[z].buildClassifier(binTrainDataset);
				
				testNaive(test,z,genres);
				testSMO(test,z,genres);
				testRandomForest(test,z,genres);

			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();

		}

	}
	

	public static void testNaive(Instances test, int z, List<String> genres ) {
		try {
			Instances unlabeled = new Instances (test);
			for (int i = 0; i < test.numInstances();i++){
				unlabeled.instance(i).setClassMissing(); 
			}

			System.out.println("Unlabeled:\n");
			System.out.println(unlabeled);				

			//Classifying unlabeled instances
			System.out.println("\nClassifying instances:\n");

			for (int i = 0; i < unlabeled.numInstances();i++){
				System.out.print("Instance ");
				System.out.print(i);


				String predicted;
				String expected;
				if(vettNaive[z].classifyInstance(unlabeled.instance(i)) == 0)
					predicted = genres.get(z);
				else
					predicted = "other";
				expected = genres.get((int)test.instance(i).classValue());

				System.out.print("\nEstimated Class: ");
				System.out.println(predicted);
				System.out.print("Actual Class: ");
				System.out.println(expected);

				if (predicted.equals(expected)) {
					confusionMatrix[z][z] = confusionMatrix[z][z] +1;
				} else if(!predicted.equals(expected) && !predicted.equals("other")) {
					confusionMatrix[z][(int)test.instance(i).classValue()] = confusionMatrix[z][(int)test.instance(i).classValue()] +1;
				}
			}	
		///printmatrix
	
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
}


	public static void testSMO(Instances test, int z, List<String> genres ) {
		try {
			Instances unlabeled = new Instances (test);
			for (int i = 0; i < test.numInstances();i++){
				unlabeled.instance(i).setClassMissing(); 
			}

			System.out.println("Unlabeled:\n");
			System.out.println(unlabeled);				

			//Classifying unlabeled instances
			System.out.println("\nClassifying instances:\n");

			for (int i = 0; i < unlabeled.numInstances();i++){
				System.out.print("Instance ");
				System.out.print(i);


				String predicted;
				String expected;
				if(vettSMO[z].classifyInstance(unlabeled.instance(i)) == 0)
					predicted = genres.get(z);
				else
					predicted = "other";
				expected = genres.get((int)test.instance(i).classValue());

				System.out.print("\nEstimated Class: ");
				System.out.println(predicted);
				System.out.print("Actual Class: ");
				System.out.println(expected);

				if (predicted.equals(expected)) {
					confusionMatrixSMO[z][z] = confusionMatrixSMO[z][z] +1;
				} else if(!predicted.equals(expected) && !predicted.equals("other")) {
					confusionMatrixSMO[z][(int)test.instance(i).classValue()] = confusionMatrixSMO[z][(int)test.instance(i).classValue()] +1;
				}
			}	
		
	
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
			
	}

	public static void testRandomForest(Instances test, int z, List<String> genres ) {
		try {
			Instances unlabeled = new Instances (test);
			for (int i = 0; i < test.numInstances();i++){
				unlabeled.instance(i).setClassMissing(); 
			}

			System.out.println("Unlabeled:\n");
			System.out.println(unlabeled);				

			//Classifying unlabeled instances
			System.out.println("\nClassifying instances:\n");

			for (int i = 0; i < unlabeled.numInstances();i++){
				System.out.print("Instance ");
				System.out.print(i);


				String predicted;
				String expected;
				if(vettRandomForest[z].classifyInstance(unlabeled.instance(i)) == 0)
					predicted = genres.get(z);
				else
					predicted = "other";
				expected = genres.get((int)test.instance(i).classValue());

				System.out.print("\nEstimated Class: ");
				System.out.println(predicted);
				System.out.print("Actual Class: ");
				System.out.println(expected);

				if (predicted.equals(expected)) {
					confusionMatrixRF[z][z] = confusionMatrixRF[z][z] +1;
				} else if(!predicted.equals(expected) && !predicted.equals("other")) {
					confusionMatrixRF[z][(int)test.instance(i).classValue()] = confusionMatrixRF[z][(int)test.instance(i).classValue()] +1;
				}
			}	
		
	
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
	}
	
}