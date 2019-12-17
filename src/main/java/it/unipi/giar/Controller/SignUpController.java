package it.unipi.giar.Controller;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;

import javax.xml.bind.DatatypeConverter;

import com.jfoenix.controls.JFXButton;
import com.jfoenix.controls.JFXComboBox;
import com.jfoenix.controls.JFXPasswordField;
import com.jfoenix.controls.JFXTextField;

import it.unipi.giar.Data.User;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.input.KeyEvent;
import javafx.scene.text.Text;
import javafx.stage.Stage;

public class SignUpController  {
	
	private boolean errorFlag = false;
	
    @FXML
    private JFXButton signUpButton;

    @FXML
    private JFXTextField signUpNickname;

    @FXML
    private JFXPasswordField signUpPassword;

    @FXML
    private JFXTextField signUpEmail;

    @FXML
    private JFXPasswordField signUpConfirmPassword;

    @FXML
    private JFXComboBox<String> signUpCountry;
    
    @FXML
    private Text errorMessage;
    
    public void initialize() {
    	try {
			List<String> countries1 = Files.readAllLines(new File("src/main/resources/countries.txt").toPath(), Charset.defaultCharset());
			ObservableList<String> countries = FXCollections.observableArrayList(countries1);
			signUpCountry.setItems(countries);
    	} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
    	}	
    }
    
    @FXML
    void checkNickname(KeyEvent event) {
    	if(!User.checkNickname(signUpNickname.getText())) {
    		errorMessage.setText("Nickname already exists.");
    		errorMessage.setVisible(true); errorFlag = true; 
    	} else {
    		errorMessage.setVisible(false); errorFlag = false; 
    	} 
    }

    @FXML
    void checkEmail(KeyEvent event) {
    	if(!User.checkEmail(signUpEmail.getText())) {
    		errorMessage.setText("Email already exists.");
			errorMessage.setVisible(true);
			errorFlag = true;
    	} else {
    		errorMessage.setVisible(false);
    		errorFlag = false;
    	}
    }

    @FXML
    void checkPassword(KeyEvent event) {
		String password = signUpPassword.getText();
		String confirmPassword = signUpConfirmPassword.getText();

		if(!password.equals(confirmPassword)) {
			errorMessage.setText("Password doesn't match.");
			errorMessage.setVisible(true);
			errorFlag = true;
		} else {
			errorMessage.setVisible(false);
			errorFlag = false;
		}
    }

    @FXML
    void SignUp(ActionEvent event) {
	    if (!errorFlag) {
	    	errorMessage.setVisible(false);
	    	
			MessageDigest md;
			try {
				md = MessageDigest.getInstance("MD5");
				md.update(signUpPassword.getText().getBytes());
				byte[] digest = md.digest();
				String password = DatatypeConverter.printHexBinary(digest).toUpperCase();
				String nickname = signUpNickname.getText();
				String email = signUpEmail.getText();
				String country = signUpCountry.getValue();
				
				User user = new User("player", nickname, email, password, country);
				user.register();
			} catch (NoSuchAlgorithmException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			try {
				loadLogin();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		} else {
			errorMessage.setText("Unable to register.");
			errorMessage.setVisible(true);
			errorFlag = true;
		}
    }
    
    void loadLogin() throws IOException {
    	Parent root = FXMLLoader.load(getClass().getResource("/fxml/SignIn.fxml"));
        Stage stage = new Stage();
        stage.setTitle("GIAR");
        stage.setScene(new Scene(root));  
        stage.show();
        stage.setResizable(false);
        Stage stage1 = (Stage)signUpButton.getScene().getWindow();
        stage1.close();
    }

}
