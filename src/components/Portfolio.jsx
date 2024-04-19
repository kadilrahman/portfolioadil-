import React from 'react';
import MDE from "../imports/MDE.jpeg";
import NLP from "../imports/eng-fra.png";
import FlightPred from "../imports/flight.jpeg"
import Diabetes from "../imports/diabetes.png"
import SpotReg from "../imports/reg.png"
import SpotClass from "../imports/class.png"
import emo from "../imports/emotion.png"
import goodread from "../imports/goodread.png"
import color from "../imports/color.png"
import barber from "../imports/barber.png"
import ecommerce from "../imports/ecommerce.jpeg"
import portfolio from "../imports/portfolio.png"
import recommender from "../imports/recommend.png"



import { useState } from 'react';

const InfoModal = ({ isOpen, onClose, title, ReadMe }) => {
  if (!isOpen) return null;

  const renderContent = (contentArray) => contentArray.map((item, index) => {
    if (item.type === "h2") {
      return <h2 key={index} className="text-lg font-bold mt-4">{item.content}</h2>;
    } else if (item.type === "p") {
      return <p key={index} className="mt-2">{item.content}</p>;
    }
    return null;
  });

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center p-4 z-50 overflow-y-auto" onClick={onClose}>
      <div className="bg-gray-800 text-white max-w-md w-full rounded-lg" onClick={(e) => e.stopPropagation()}>
        <div className="p-4 overflow-y-auto max-h-[80vh]"> {/* Make this div scrollable */}
          <h3 className="text-xl font-bold">{title}</h3>
          {ReadMe && renderContent(ReadMe)}
          <button onClick={onClose} className="mt-4 px-4 py-2 bg-green-500 rounded hover:bg-green-700 transition-colors">Close</button>
        </div>
      </div>
    </div>
  );
};
export const Portfolio = () => {

  const [selectedProject, setSelectedProject] = useState(null);

  // Function to open modal with project info
  const openModal = (project) => {
    setSelectedProject(project);
  };

  // Function to close modal
  const closeModal = () => {
    setSelectedProject(null);
  };

    const portfolios = [
      {
        id: 1,
        src: MDE,
        title: "Monocular Depth Estimation",
        info: "In this project, I delved into depth estimation, a challenging aspect of Computer Vision, focusing on both outdoor and indoor environments. Through extensive research, including reviewing state-of-the-art (SOTA) methodologies, I gained insights into the specific challenges of outdoor depth estimation and the necessary pre-processing steps. I implemented a fully convolutional residual network, achieving results comparable to SOTA models. This experience broadened my knowledge in Computer Vision, covering data cleansing, pre-processing, model development, hyperparameter tuning, and evaluation. The project not only enhanced my technical skills but also provided a comprehensive understanding of the depth estimation process.",
        githubUrl: "https://github.com/kadilrahman/Depth-estimation-FCRN-.git",
        ReadMe: [
          { type: "p", content: "Monocular depth estimation is an important task in computer vision that aims to predict three-dimensional depth information from a single two-dimensional image. A long-standing challenge in computer vision has been monocular depth estimation, which has critical implications for robotics, autonomous driving, and augmented reality. The dissertation examines the application and evaluation of Fully Convolutional Residual Networks (FCRN) in monocular depth estimation, a critical task in computer vision involving predicting the depth of a scene based on one input image. Providing a realistic but computationally manageable benchmark for the experiments, the study uses a subset of a larger dataset. The FCRN model exhibits promising results, demonstrating a competitive level of accuracy in depth prediction across diverse environmental settings despite not achieving state-of-the-art performance. This research contributes to ongoing efforts in computer vision by shedding light on the capabilities and limitations of FCRN architectures for monocular depth estimation tasks and suggests avenues for future improvements. Although the dissertation's primary focus remains on the application of the FCRN model, it also examines the inherent challenges associated with monocular depth estimation. Using multiple viewpoints or special sensors is often necessary to gauge depth, but the goal is to do so with a single RGB image, increasing the system's utility and applicability. By training and evaluating the model on a subset of the KITTI dataset, a high-quality dataset representative of real-world situations is used. Although the proposed FCRN model performs at a level close to state-of-the-art, it also highlights the room for improvement and sets the stage for future research to enhance its predictive accuracy and computational efficiency." },
          { type: "h2", content: "Usage" },
          { type: "p", content: "In order to run the code, simply open the google colab file and run all cells. Please note that all the dataset files are downloaded from google drive. In order to inpaint the images please follow the process mentioned in the notebook. The inpainting process requires alot of GPU and time. So its better to use local system (if the GPU and processor is good) in order to perform inpainting on the kitti dataset." },
          { type: "h2", content: "Dataset" },
          { type: "p", content: "Machine learning models need to be trained and evaluated using an appropriate dataset, especially monocular depth estimation models. The KITTI (Karlsruhe Institute of Technology and Toyota Technological Institute) dataset has been selected as the primary resource for this study. The KITTI dataset is a benchmark suite specifically designed to advance the field of autonomous driving, offering a rich set of high-quality data samples that are pertinent to various computer vision tasks. https://www.cvlibs.net/datasets/kitti/" },
          { type: "h2", content: "Inpainting" },
          { type: "p", content: "The annotated depth data acquired from KITTI dataset is sparse point clouds, unlike the dense depth maps its missing few pixel. In order to convert sparse point clouds into complete depth maps the framework performs inpainting or filling. Inpainting algorithms to fill in any missing or noisy depth information are applied by using a guided filter-based inpainting technique that considers both depth and RGB images. The in-painted depth maps aim to remove noise and improve the quality of depth data. The method of inpainting was inspired by Alhashim el al. code in order to perform inpainting on both NYU and KITTI dataset. https://arxiv.org/abs/1812.11941" },

        ],
      },

      {
        id: 2,
        src: NLP,
        title: "Eng-Fra NLP",
        info: "This code implements a neural machine translation model using TensorFlow to translate English sentences into French. It leverages a sequence-to-sequence architecture with attention mechanisms, ensuring the model focuses on relevant parts of the sentence during translation. The process involves text normalization, tokenization, and embedding, followed by encoding English sentences and decoding them into French. The model uses GRU layers for both the encoder and decoder, with a custom attention layer enhancing translation quality. This project demonstrates a deep understanding of NLP principles and TensorFlow's capabilities in handling complex sequence data, showcasing the potential for advanced language translation solutions.",
        githubUrl: "https://github.com/kadilrahman/NLP-english-french.git",
        ReadMe: [
          { type: "h2", content: "Objective" },
          { type: "p", content: "The goal is to develop a machine learning model capable of translating English sentences into French. This involves natural language processing (NLP) techniques and deep learning models to understand the semantics of the English language and generate corresponding French translations." },
          { type: "h2", content: "Dataset" },
          { type: "p", content: "The dataset consists of English and French sentence pairs. It is loaded and preprocessed to split the sentences into words and convert them into numerical representations that can be fed into a neural network. The dataset is extracted from a ZIP file containing a text file (fra.txt) with English-French sentence pairs." },
          { type: "h2", content: "Approach" },
          { type: "p", content: "Data Preprocessing: The English and French sentences are extracted, tokenized, and converted into a format suitable for training. This includes lowercasing, removing special characters, and adding start and end tokens to sentences." },
          { type: "h2", content: "Model Building" },
          { type: "p", content: "Text Vectorization: Converts text inputs into integer sequences. This is done for both the source (English) and target (French) languages." },
          { type: "p", content: "Encoder-Decoder Architecture: The core of the model, consisting of an encoder that processes the input sentences and a decoder that generates the translated sentences. The Encoder uses a GRU (Gated Recurrent Unit) layer to process the input sequences. The Decoder also uses a GRU layer and incorporates an attention mechanism to focus on different parts of the input sequence when generating each word of the output sequence." },
          { type: "p", content: "Attention Mechanism: Allows the model to weigh the importance of different input words for each word in the output sentence, improving the quality of the translation." },
          { type: "p", content: "Training and Evaluation: The model is trained on the preprocessed dataset and evaluated based on its ability to translate new sentences from English to French." },
          { type: "h2", content: "Key Components and Libraries" },
          { type: "p", content: "TensorFlow & TensorFlow Text: For building and training the deep learning model and preprocessing text data." },
          { type: "p", content: "Einops: Used for more readable tensor operations." },
          { type: "p", content: "Numpy & Matplotlib: For data manipulation and visualization." },
          { type: "h2", content: "Code Highlights" },
          { type: "p", content: "The code includes custom layers and functions for preprocessing text, building the seq2seq model with attention, and utilities for converting between text and tokens. The use of TensorFlow's TextVectorization layer for preparing text data for the model. Implementation of a custom ShapeChecker class to ensure tensor shapes are as expected throughout the model, aiding in debugging. Detailed steps for encoding input sequences, applying attention, and decoding to generate translations, showcasing advanced TensorFlow techniques. This project exemplifies a sophisticated application of deep learning to NLP, demonstrating the power of seq2seq models with attention for language translation tasks." },

        ],
      },
      {
        id: 3,
        src: FlightPred,
        title: "Flight Price Prediction",
        info: "This project develops a machine learning model to predict flight prices, employing Python libraries like Pandas for data manipulation and Scikit-learn for modeling. Initial steps include data cleaning to handle missing values and dropping irrelevant features, followed by feature engineering to extract useful information from dates and times. The model uses one-hot encoding to handle categorical variables, making the data suitable for the RandomForestRegressor algorithm. The performance of the trained model is evaluated using mean squared error (MSE) and root mean squared error (RMSE), showcasing the model's ability to accurately predict flight prices based on historical data, demonstrating a practical application of data science in the travel industry.",
        githubUrl: "https://github.com/kadilrahman/Flight-Prediction-.git",
        ReadMe: [
          { type: "h2", content: "Objective" },
          { type: "p", content: "The goal of this project is to develop a machine learning model capable of predicting flight prices based on historical data. The challenge involves handling a dataset that includes various features related to flights, making it a regression problem since the output variable (flight price) is continuous." },
          
          { type: "h2", content: "Dataset" },
          { type: "p", content: "The dataset comprises two Excel files: Data_Train.xlsx for training and Test_set.xlsx for testing. These files contain records of flight details, including airlines, departure and arrival times, and prices. Key features include:" },
          { type: "p", content: "Airline: The flight's operating airline." },
          { type: "p", content: "Date_of_Journey: The flight's departure date." },
          { type: "p", content: "Source: The departure location." },
          { type: "p", content: "Destination: The arrival location." },
          { type: "p", content: "Duration: Flight duration." },
          { type: "p", content: "Total_Stops: The number of stops before reaching the destination." },
          { type: "p", content: "Price: The cost of the flight (target variable)." },

          { type: "h2", content: "Approach" },
          { type: "p", content: "The approach to solving this regression problem involves several steps:" },
          { type: "p", content: "Data Cleaning: Identifying and handling missing values, removing unnecessary columns, and correcting inconsistencies in categorical data." },
          { type: "p", content: "Feature Engineering: Extracting useful information from existing features, such as converting the 'Duration' from hours and minutes to minutes only, and splitting 'Date_of_Journey' into day, month, and weekday." },
          { type: "p", content: "Data Preprocessing: Standardizing the case of categorical variables to avoid duplicates, and handling categorical variables through encoding." },
          { type: "p", content: "Model Development: Splitting the data into training and testing sets, selecting appropriate regression models, and tuning hyperparameters for optimal performance." },
          { type: "p", content: "Evaluation: Using metrics like mean absolute error and mean squared error to assess model performance." },

          { type: "h2", content: "Required Python Libraries" },
          { type: "p", content: "Pandas and Numpy for data manipulation." },
          { type: "p", content: "Matplotlib and Seaborn for data visualization" },

          { type: "p", content: "Scikit-Learn for model selection, training, and evaluation." },

          { type: "p", content: "The project emphasizes the importance of thorough data preprocessing and feature engineering to improve model accuracy and performance in predicting flight prices." },


        ],
      },
      {
        id: 4,
        src: Diabetes,
        title: "Diabetes Prediction",
        info: "This project involves developing a predictive model for diabetes using a Support Vector Machine (SVM) algorithm, showcasing the application of machine learning in healthcare. The dataset is initially analyzed to understand the difference in mean values of features between diabetic and non-diabetic individuals, visualized through bar charts for clear comparison. The data is then preprocessed, including scaling features for optimal performance of the SVM model. The model is trained and evaluated on a split dataset, demonstrating its ability to accurately classify individuals based on their risk of diabetes. The project highlights the effectiveness of SVM in binary classification problems, achieving a notable accuracy score on test data, underscoring the potential of machine learning in enhancing diagnostic processes and personalized medicine strategies.",
        githubUrl: "https://github.com/kadilrahman/Diabetes-Prediction.git",
        ReadMe: [
          { type: "h2", content: "Objective" },
  
          { type: "p", content: "This project aims to predict the presence of diabetes in individuals using a Support Vector Machine (SVM) model, a powerful tool for classification tasks. The dataset, sourced from a medical study, includes various health metrics such as glucose levels, blood pressure, and insulin levels, labeled with outcomes indicating the presence or absence of diabetes." },
          { type: "h2", content: "Dataset" },
          { type: "p", content: "The dataset, diabetes.csv, comprises several features that are crucial for predicting diabetes outcomes. These features include patient health metrics like pregnancies, glucose concentration, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, and age. The 'Outcome' column indicates whether the patient has diabetes, making this a binary classification problem." },
          { type: "h2", content: "Approach" },
          { type: "p", content: "The analysis begins with data exploration, where feature mean values are compared between individuals with and without diabetes. This comparison helps identify significant factors contributing to diabetes. A visual representation of these means and their percentage differences highlights the impact of each feature on diabetes outcomes. Data preprocessing involves standardizing the features to have a mean of zero and a standard deviation of one. This ensures that the SVM model's performance is not biased by the data scale. The dataset is then split into training and testing sets to evaluate the model's performance accurately. The SVM model, with a linear kernel, is trained on the dataset. The choice of a linear kernel is due to its effectiveness in high-dimensional spaces, typical of medical datasets. Model performance is assessed based on accuracy scores for both training and testing sets, providing insights into the model's ability to generalize to new data." },
          { type: "h2", content: "Required Python libraries" },
          { type: "p", content: "NumPy: For numerical operations on large, multidimensional arrays and matrices." },
          { type: "p", content: "Matplotlib & Seaborn: For data visualization, enabling the graphical representation of data analysis findings." },
          { type: "p", content: "Scikit-Learn: For implementing machine learning algorithms, particularly SVM for this project, and for data preprocessing and model evaluation tools like train_test_split and accuracy_score." },
          { type: "p", content: "This project exemplifies the application of machine learning in healthcare, demonstrating how SVM can be utilized to predict diabetes. This can aid in early diagnosis and management." },
        ],
      },
      {
        id: 5,
        src: SpotReg,
        title: "Regression on Spotify Data",
        info: "The objective of the challenge is to build a machine learning model that is able to predict the popularity score of a song. Popularity of a song is a continuous numerical value and this problem statement is a regression problem. The approach is towards solving this problem is by using various regression models like Catboost Regressor, Stochastic Gradient descent Regressor, Support Vector Regression, Decision Tree, Random Forest Regressor and Gradientboost Regressor so as to choose the regression model showcasing optimum efficiency. In order to progress towrads this ultimate output, we have performed various preliminary operations on the datasets which include cleaning the datasets and visualizing relationships between various features and target variable (i.e Popularity) using Exploratory Data Analysis.",
        githubUrl: "https://github.com/kadilrahman/spotify-regression.git",
        ReadMe: [
          { type: "h2", content: "Objective" },
          { type: "p", content: "The objective of the challenge is to build a machine learning model that is able to predict the popularity score of a song. Popularity of a song is a continuous numerical value and this problem statement is a regression problem." },
          { type: "h2", content: "Dataset" },
          { type: "p", content: "There are two files CS98XRegressionTrain.csv that contains the training dataset and CS98XRegressionTest.csv that contains the dataset for final testing. The dataset is a collection of spotify songs with their audio features (tempo, energy, danceability etc.) The training dataset contains 15 columns that are described below:" },
          { type: "p", content: "Id - an arbitrary unique track identifier" },
          { type: "p", content: "title - track title" },
          { type: "p", content: "artist - singer or band" },
          { type: "p", content: "top genre - genre of the track" },
          { type: "p", content: "year - year of release (or re-release)" },
          { type: "p", content: "bpm - beats per minute (tempo)" },
          { type: "p", content: "nrgy - energy: the higher the value the more energetic" },
          { type: "p", content: "dnce - danceability: the higher the value, the easier it is to dance to this song" },
          { type: "p", content: "dB - loudness (dB): the higher the value, the louder the song" },
          { type: "p", content: "live - liveness: the higher the value, the more likely the song is a live recording" },
          { type: "p", content: "val - valence: the higher the value, the more positive mood for the song" },
          { type: "p", content: "dur - duration: the length of the song" },
          { type: "p", content: "acous - acousticness: the higher the value the more acoustic the song is" },
          { type: "p", content: "spch - speechiness: the higher the value the more spoken word the song contains" },
          { type: "p", content: "pop - popularity: the higher the value the more popular the song is (and the target variable for this problem)" },

          { type: "h2", content: "Approach" },
          { type: "p", content: "The approach of our group towards solving this problem is by using various regression models like Catboost Regressor, Stochastic Gradient descent Regressor, Support Vector Regression, Decision Tree, Random Forest Regressor and Gradientboost Regressor so as to choose the regression model showcasing optimum efficiency. In order to progress towrads this ultimate output, we have performed various preliminary operations on the datasets which include cleaning the datasets and visualizing relationships between various features and target variable (i.e Popularity) using Exploratory Data Analysis." },
          { type: "h2", content: "Required python libraries" },
          { type: "p", content: "Numpy - for numerical operations" },
          { type: "p", content: "Pandas - for loading, querying and manipulating datasets" },
          { type: "p", content: "Matplotlib (pyplot) - for visual analysis" },
          { type: "p", content: "Seaborn - for visual analysis" },
          { type: "p", content: "Scikit-Learn - for machine learning" },
          { type: "p", content: "Catboost - for CatBoost Regressor Model" },
        ],
      },
      {
        id: 6,
        src: SpotClass,
        title: "Classification on Spotify Data",
        info: "The objective of the challenge is to build a machine learning model that is able to predict the genre of a song. Using data provided by Spotify, Genre of a song is a categorical value and this problem statement is a classification problem. There are various possible genres that makes this a multinomial classification problem. In tackling a classification problem within machine learning, the process begins with Data Analysis to understand dataset specifics, followed by Data Preparation & Feature Engineering to clean and enhance data. Identifying the data distribution and detecting outliers help refine the dataset for accurate modeling. Feature Selection is crucial to pinpoint relevant predictors. Preprocessing steps like Scaling and Encoding standardize the dataset, while Train/Test Split ensures a reliable evaluation framework. ",
        githubUrl: "https://github.com/kadilrahman/spotify-classification.git",
        ReadMe: [
          { type: "h2", content: "Objective" },
          { type: "p", content: "Objective of the challenge is to build a machine learning model that is able to predict the genre of a song. Using data provided by Spotify, Genre of a song is a categorical value and this problem statement is a classification problem. There are various possible genres that makes this a multinomial classification problem." },
          { type: "h2", content: "Dataset" },
          { type: "p", content: "There are two files CS98XClassificationTrain.csv that contains the training dataset and CS98XClassificationTest.csv that contains the dataset for final testing. These files have been provided in the repository. The dataset is a collection of spotify songs with their audio features (tempo, energy, danceability etc.) The training dataset contains 15 columns that are described below:" },
          { type: "p", content: "Id - an arbitrary unique track identifier" },
          { type: "p", content: "title - track title" },
          { type: "p", content: "artist - singer or band" },
          { type: "p", content: "year - year of release (or re-release)" },
          { type: "p", content: "bpm - beats per minute (tempo)" },
          { type: "p", content: "nrgy - energy: the higher the value the more energetic" },
          { type: "p", content: "dnce - danceability: the higher the value, the easier it is to dance to this song" },
          { type: "p", content: "dB - loudness (dB): the higher the value, the louder the song" },
          { type: "p", content: "live - liveness: the higher the value, the more likely the song is a live recording" },
          { type: "p", content: "val - valence: the higher the value, the more positive mood for the song" },
          { type: "p", content: "dur - duration: the length of the song" },
          { type: "p", content: "acous - acousticness: the higher the value the more acoustic the song is" },
          { type: "p", content: "spch - speechiness: the higher the value the more spoken word the song contains" },
          { type: "p", content: "pop - popularity: the higher the value the more popular the song is" },
          { type: "p", content: "top genre - genre of the track (and the target variable for this problem)" },


          { type: "h2", content: "Approach" },
          { type: "p", content: "As for any machine learning project, we will be following a series of steps to come up with the solution to our classification problem." },
          { type: "p", content: "Data Analysi" },
          { type: "p", content: "Data Preparation & Feature Engineering:- dentifying Data distribution, Outlier detection, Feature selection, Preprocessing" },
          { type: "p", content: "Model Selection:- basic models" },
          { type: "p", content: "Model training & evaluation" },
          { type: "p", content: "Ensemble learning" },

          { type: "h2", content: "Required python libraries" },
          { type: "p", content: "Numpy - for numerical operations" },
          { type: "p", content: "Pandas - for loading, querying and manipulating datasets" },
          { type: "p", content: "Matplotlib (pyplot) - for visual analysis" },
          { type: "p", content: "Seaborn - for visual analysis" },
          { type: "p", content: "Scikit-Learn - for machine learning" },

        ],
      },

      {
        id: 7,
        src: emo,
        title: "Emotion Recognition",
        info: "As a group, we performed the Emotion Recognition Task using Random Forest Classifier, Deep Neural Networks Model, Convolutional Neural Networks (CNN), Residual Network Model (ResNet), Convolutional Recurrent Neural Network Model (CRNN), and Transformer Model. One of the main challenges we faced is the lack of standard datasets and benchmarks along with the huge size of the dataset which made it hard for us to process it and execute the algorithms. In terms of recommendations, we would highly recommend carefully preprocess the data and select appropriate features that capture the relevant emotional cues. We also find that it is important to tune in appropriate parameters/hyperparameters so as to obtain optimum performance.",
        githubUrl: "https://github.com/kadilrahman/emotion_recognition.git",
        ReadMe: [
          { type: "h2", content: "Objective" },
          { type: "p", content: "We performed the Emotion Recognition Task using Random Forest Classifier, Deep Neural Networks Model, Convolutional Neural Networks (CNN), Residual Network Model (ResNet), Convolutional Recurrent Neural Network Model (CRNN), and Transformer Model. One of the main challenges we faced is the lack of standard datasets and benchmarks along with the huge size of the dataset which made it hard for us to process it and execute the algorithms. In terms of recommendations, we would highly recommend carefully preprocess the data and select appropriate features that capture the relevant emotional cues. We also find that it is important to tune in appropriate parameters/hyperparameters so as to obtain optimum performance." },
          { type: "h2", content: "Method" },
          { type: "p", content: "The given dataset primarily consists of two sub-datasets namely training and testing. The training dataset consists of 3 features namely “id”, “emotion” and “pixels” wherein “id” represents the index number, “emotion” consists of all the types of emotions namely Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral and “pixels” representing the data pertaining to the respective emotions. The testing dataset consists of primarily 2 features namely “id” and “pixels” which are used as an input to predict the “emotion” of the input variables. Given this data, we learnt that it is essential to carefully preprocess the data so as to obtain optimum efficiency." },
          

          { type: "h2", content: "Pre-processing" },
          { type: "p", content: "For pre-processing, we firstly check if there are any missing/null values in the dataset post which we prepared the pixel data from training and testing data frames to be used as an input for the ML models. Secondly, for train, test and validation, we have only split the training dataset into training and validation sets. Additionally, for pre-processing the data, we have split the “pixels” feature and reshaped the input images in the size 48x48 and we have also reshaped this feature based on every model’s requirements so as to better train the models and obtain the desired output. We have also used One-hot Encoding in some of the models as it was necessary for the model to improve its performance." },
          { type: "h2", content: "Data augmentation" },
          { type: "p", content: "Moving on, we have implemented data augmentation for our preprocessing which we executed by setting up an Image Data Generator object with various data augmentation parameters which we used to generate augmented images for model training." },
          { type: "h2", content: "Models" },
          { type: "p", content: "Standard ML Baseline: Random Forest Classifier:" },
          { type: "p", content: "Simple Machine Learning Baseline Model: We have used Random Forest Classifier for this task. Random Forest Classifier is a popular machine learning algorithm that is well-suited for classification tasks. We have chosen it mainly due to its ability to handle complex relationships, robustness to noise and outliers, feature importance estimation, and ensemble learning capabilities." },
          { type: "p", content: "Deep NN model and CNN" },
          { type: "p", content: "We have used Convolutional Neural Network as our Deep Neural Network Model. It is a type of neural network commonly used for image and video classification tasks. We chose it mainly due to its ability to handle complex relationships between input features and output labels, and their high accuracy in a wide range of image and speech recognition tasks." },
          { type: "p", content: "Residual Network (ResNet):" },
          { type: "p", content: "ResNet is a deep neural network which uses skip connections that allow the network to learn residual functions instead of directly learning the mapping between input and output. We have mainly considered to use it due to its ability to show excellence in image recognition tasks. One of its cardinal strength is to effectively train deep neural networks with hundreds of layers by addressing the problem of vanishing gradients that can occur during training." },
          { type: "h2", content: "Results" },
          { type: "p", content: "According to our results, we can conclude that Convolutional Recurrent Neural Networks Model (CRNN) is the best performing model having an accuracy of 0.69. We achieved such a high performance primarily because of Data Augmentation. We have used data augmentation in this model which helps in increasing the diversity of training data, improving the generalization of the model as well as in producing efficient and robust models. We have computed the mean and standard deviations for all the models used which include Random Forest Classifier Model, Deep Neural Networks Model (DNN), Convolutional Neural Networks Model (CNN), Transformer Model, Convolutional Recurrent Neural Networks Model (CRNN) and Residual Networks Model (ResNet). The average mean and standard deviation for our results are 0.18and 0.09 respectively and the mean and standard deviation of our best model (i.e CRNN Model) are 0.16 and 0.23 respectively. Secondly, the Convolutional Neural Networks Model (CNN) and the Transformer model are the worst performing models as per the above table. This can be seen primarily due to not using data augmentation for each of these models. It can also be seen that the epoch input also plays a cardinal role in the success of the models. In this case, for CNN and Transformer, we have kept a significantly low epoch value for the purpose of training the model which thereby implies the low accuracy score. We tried different epoch values for our best model so as to check for the optimum result however, due to technical limitations we were unable to attain the desired value for the epochs used. We have set epochs as 200 for the CRNN model, which is our best model, and was the highest which could work using the available resources we had at our disposal." },
          { type: "h2", content: "Summary" },
          { type: "p", content: "We would recommend using the Convolutional Recurrent Neural Networks Model (CRNN) for this task as the model that performed the best was CRNN achieving an accuracy of 0.69. Due to CRNNs features such as capturing of temporal and spatial information, robustness to noise, flexibility and state-of-the-art performance makes CRNN the best model to use. Data Augmentation also plays a vital role in the success of this model which is primarily responsible for improving the performance and robustness of a CRNN model. It not only improves generalization performance but also increases data efficiency in training the CRNN model. Secondly, as stated above, epoch values also have a significant impact on the accuracy of a model. Having used an epochs value of 200 for the CRNN model, makes CRNN a better performer than the rest. Post submitting the models on Kaggle, we received a score of 0.65, which can be viewed as a good performance especially in comparison with peer groups. The score has some room for improvement whatsoever and can be done further by increasing the epochs value. In our case we could not achieve it due to technical inability however, given a favourable technical infrastructure, we could attain a better overall score." },

        ],
      },

      {
        id: 8,
        src: goodread,
        title: "Goodread Rating Prediction",
        info: "My main task was to predict the rating of a random book, which I achieved by implementing the appropriate machine learning algorithms. I built three models: a Baseline Machine Learning Model where I used a Logistic Regression model, a 3-layer Neural Network Model, and a Deep Neural Network (DNN). For the Complex Neural Network, I used Long Short-Term Memory (LSTMs), which is a type of Recurrent Neural Network (RNN), Gated Recurrent Units (GRUs) which are again a type of Recurrent Neural Networks (RNN) used for modeling sequential data, and Bidirectional Encoder Representations from Transformers (BERT). BERT uses a bidirectional approach to generate contextualized word embeddings, which are representations of words that capture their meanings in the context of a sentence. This approach is beneficial because it is pre-trained on massive amounts of text data, allowing it to learn general language patterns and relationships between words.",
        githubUrl: "https://github.com/kadilrahman/goodread_prediction-.git",
        ReadMe: [
          { type: "h2", content: "Objective" },
          { type: "p", content: "My main task was to predict the rating of a random book, which I achieved by implementing the appropriate machine learning algorithms. I built three models: a Baseline Machine Learning Model where I used a Logistic Regression model, a 3-layer Neural Network Model, and a Deep Neural Network (DNN). For the Complex Neural Network, I used Long Short-Term Memory (LSTMs), which is a type of Recurrent Neural Network (RNN), Gated Recurrent Units (GRUs) which are again a type of Recurrent Neural Networks (RNN) used for modeling sequential data, and Bidirectional Encoder Representations from Transformers (BERT). BERT uses a bidirectional approach to generate contextualized word embeddings, which are representations of words that capture their meanings in the context of a sentence. This approach is beneficial because it is pre-trained on massive amounts of text data, allowing it to learn general language patterns and relationships between words." },
          { type: "h2", content: "Clean review_text" },
          { type: "p", content: "I cleaned the review text and removed the noise caused by abundant punctuations, wide spaces, and invalid characters, which would otherwise have been responsible for the inefficiency in executing the algorithms I used. It's vital to get rid of unnecessary input values and classify nouns, pronouns, and verbs. I split the data into training, testing, and validation according to the best fit and kept it consistent to avoid void/inefficient results and achieve optimum efficiency." },
          { type: "h2", content: "Model" },
          { type: "p", content: "Standard ML Baseline: Logistic regression" },
          { type: "p", content: "For my standard machine learning baseline model, I chose to use the Logistic Regression Model. It's a widely used statistical method for binary classification tasks, where the goal is to predict a binary outcome, such as whether a book is highly rated or not. I believe it can be a useful method for classification tasks, and its performance can be easily assessed using performance metrics like the F1 Score, among others." },
          { type: "p", content: "3Layer NN Baseline" },
          { type: "p", content: "My 3-Layer Neural Network Model consists of an input layer, a hidden layer, and an output layer, where each layer contains one or more neurons. These neurons are computational units that perform mathematical operations on the input data. Given the necessity to experiment with different parameters for the task at hand, I find this model to be the best fit for such tasks." },
          { type: "p", content: "Deep NN" },
          { type: "p", content: "For my deep neural networks model, I decided to create a 7-layer DNN model that includes 3 base layers of neurons. I explored the optimum number of layers needed and concluded that adding more layers would incrementally improve performance. Based on my best result, which was achieved with 15 layers and 55 neurons (the best configuration from the initial 3 layers), I tested various activation functions like relu, sigmoid, tanh, and elu. I did this to gain a better understanding of the model and to enhance its performance." },
          { type: "p", content: "Complex Neural Network Models:" },
          { type: "p", content: "I have prominently used 3 Complex Neural Network Models namely, Recurrent Neural Network Model (RNN), Long Short-Term Models (LSTMs) and Bidirectional Encoder Representations from Transformer (BERT)." },
          { type: "p", content: "RNN" },
          { type: "p", content: "Recurrent Neural Network Model (RNN): it is a type of neural network designed to handle sequential data as against the neural networks which process data in a fixed sequence of layers. I found it difficult to find the parameters to tune the model however, I was successful in generating a decent output for the same." },
          { type: "p", content: "GRU" },
          { type: "p", content: "I've been working with Gated Recurrent Units (GRUs), which are essentially a variant of Recurrent Neural Networks designed to handle sequential data, like text and speech. They're similar to LSTMs because they also use gating mechanisms to control the flow of information through the network. However, in my experience, I've noticed that the performance of the model depends on several factors, such as the quantity/size of the input data and the choice of parameters/hyperparameters, among others." },
          { type: "p", content: "BERT" },
          { type: "p", content: "I used the Bidirectional Encoder Representations from Transformer (BERT), a pre-trained deep learning model, because I found it incredibly effective for language processing tasks. I chose it as I believe it's a powerful tool for handling large amounts of data. However, I also realized that it has a significant drawback: it requires substantial computational resources to train and enhance the model's performance." },
          { type: "h2", content: "Result" },
          { type: "p", content: "After performing the given task, I clearly saw that the Bidirectional Encoder Representations from Transformer Model (BERT) outperformed all the other models I used. I would highly recommend using it because of its superior ability to understand the contextual meaning of words in a sentence and its efficiency in handling large datasets. The models I would not recommend are the Recurrent Neural Networks Model (RNN) and Deep Neural Networks Model (DNN) due to their inefficient performance and their inability to effectively process my dataset. The mean and standard deviation of all the models I tested are 0.331 and 0.227, respectively. Through my approach and learnings, I discovered the importance of a model's efficiency in managing sequential datasets that are large in volume. Additionally, it's crucial that the model executes the required task well and predicts outcomes accurately. I experimented with various models, including the Logistic Regression Model, 3 Layers Neural Networks Model (CNN), Recurrent Neural Networks Model (RNN), Deep Neural Networks Model (DNN), Gated Recurrent Units (GRUs), and Bidirectional Encoder Representation from Transformer Model (BERT). Among these, I found that adjusting the number of epochs settings was challenging. I couldn't run the models as efficiently as I wanted due to Random Access Memory (RAM) and user interface limitations, particularly in Google Colab. Furthermore, I learned that increasing the number of epochs tended to improve the accuracy score. However, due to the mentioned limitations, I was unable to execute the models at the desired level of epochs, which prevented me from achieving optimum accuracy." },
          { type: "h2", content: "Summary" },
          { type: "p", content: "In my analysis, the table I reviewed showcases the model and its respective accuracy, leading me to conclude that the Bidirectional Encoder Representations from Transformers (BERT) stands out. I can explain this as follows: Firstly, BERT is incredibly adept at grasping the contextual meanings of words in sentences, from which it meticulously extracts useful features. Secondly, its capacity to process vast amounts of data allows me to efficiently train the dataset with BERT. Lastly, because BERT is pre-trained on extensive data collections, it effortlessly learns language patterns, which I can then fine-tune to perform any specific classification task. After uploading the file on Kaggle, I achieved a score of 0.5808, which is decent considering I only set the training model to run for 2 epochs. However, I believe this could be significantly improved by increasing the number of epochs. This would likely enhance the accuracy score further. Therefore, I've observed that the number of epochs is directly proportional to the accuracy score. To achieve better results in the task, setting the epochs to a higher number, like 50, could potentially increase the score by approximately 20%." },

        ],
      },

      {
        id: 9,
        src: color,
        title: "Color Switch-Android",
        info: "Color Switch is a dynamic, addictive Android game developed using Java in Android Studio. The essence of the game lies in its simplicity and the quick reflexes it demands from players. It's designed to provide an engaging and challenging experience that tests the player's timing and color recognition skills. The game features a rotator with four differently colored cones. A target color is displayed at the top of the screen, and the player's task is to match this target color with the corresponding cone color on the rotator. Players tap on the screen to rotate the cones and align the matching color with the target. Successive matches increase the player's score, with the game's pace gradually increasing to add challenge.",
        githubUrl: "https://github.com/kadilrahman/Andrioid-application-.git",
        ReadMe: [
          { type: "h2", content: "Objective" },
          { type: "p", content: "Color Switch is a dynamic, addictive Android game developed using Java in Android Studio. The essence of the game lies in its simplicity and the quick reflexes it demands from players. It's designed to provide an engaging and challenging experience that tests the player's timing and color recognition skills. The game features a rotator with four differently colored cones. A target color is displayed at the top of the screen, and the player's task is to match this target color with the corresponding cone color on the rotator. Players tap on the screen to rotate the cones and align the matching color with the target. Successive matches increase the player's score, with the game's pace gradually increasing to add challenge." },
          
        ],
      },

      {
        id: 10,
        src: barber,
        title: "Barber Website",
        info: "I accomplished the development of a website and application for a client, ensuring fulfilment of all their stated requirements. Working along with React and Java, the project included front-end and back-end programming. The applications enable online booking and reservations for a local men’s grooming shop in Varna, Bulgaria. I accomplished to satisfy client’s needs by tweaking the fullstack development. The project involved in direct communication with client and the project was planned accordingly. ",
        githubUrl: "https://github.com/kadilrahman/bb-frontend.git",
        ReadMe: [
          { type: "h2", content: "Objective" },
          { type: "p", content: "I accomplished the development of a website and application for a client, ensuring fulfilment of all their stated requirements. Working along with React and Java, the project included front-end and back-end programming. The applications enable online booking and reservations for a local men’s grooming shop in Varna, Bulgaria. I accomplished to satisfy client’s needs by tweaking the fullstack development. The project involved in direct communication with client and the project was planned accordingly." },
          
        ],
      },

      {
        id: 11,
        src: ecommerce,
        title: "Ecommerce website",
        info: "I accomplished the development of a website and application for a client, ensuring fulfilment of all their stated requirements. Working along with Javascript, HTML and CSS, the project included front-end. The project enables free flow browsing through produces with a complete functional product presentation through website. I accomplished to satisfy client’s needs by tweaking the fullstack development. The project involved in direct communication with client and the project was planned accordingly. ",
        githubUrl: "https://github.com/kadilrahman/E-commerce-Grocery.git",
        ReadMe: [
          { type: "h2", content: "Objective" },
          { type: "p", content: "I accomplished the development of a website and application for a client, ensuring fulfilment of all their stated requirements. Working along with Javascript, HTML and CSS, the project included front-end. The project enables free flow browsing through produces with a complete functional product presentation through website. I accomplished to satisfy client’s needs by tweaking the fullstack development. The project involved in direct communication with client and the project was planned accordingly." },
          
        ],
      },

      {
        id: 12,
        src: portfolio,
        title: "Portfolio website",
        info: "I accomplished the development of a website and application for representing my complete portfolio, ensuring brief description about myself and my related work. Working along with React.js and tailwind, the project included fullstack development. It's the website you are on now :)",
        githubUrl: "https://github.com/kadilrahman/E-commerce-Grocery.git",
        ReadMe: [
          { type: "h2", content: "Objective" },
          { type: "p", content: "I accomplished the development of a website and application for a client, ensuring fulfilment of all their stated requirements. Working along with Javascript, HTML and CSS, the project included front-end. The project enables free flow browsing through produces with a complete functional product presentation through website. I accomplished to satisfy client’s needs by tweaking the fullstack development. The project involved in direct communication with client and the project was planned accordingly." },
          
        ],
      },
      {
        id: 13,
        src: recommender,
        title: "Music Recommender System",
        info: "The Music Recommender System is an advanced, interactive tool designed to personalize music discovery. Leveraging the Spotify API, this system analyzes user inputs and song lyrics to recommend songs that resonate with individual tastes and preferences. It integrates machine learning algorithms to enhance accuracy and user engagement, making music discovery a deeply personalized experience.",
        githubUrl: "https://github.com/kadilrahman/recommender-spotify.git",
        ReadMe: [
          { type: "h2", content: "Objective" },
          { type: "p", content: "To develop a user-friendly web application that provides song recommendations based on user preferences." },
          { type: "p", content: "To implement machine learning algorithms for analyzing and processing song data effectively." },
          { type: "p", content: "To integrate the Spotify API to fetch real-time data and ensure a broad database of songs." },
          { type: "p", content: "To improve music discovery by utilizing natural language processing on song lyrics for better matching." },
          { type: "h2", content: "Technologies Used" },
          { type: "p", content: "Python: For backend logic and machine learning computations." },
          { type: "p", content: "Flask: As the web framework to handle HTTP requests and serve the web pages." },
          { type: "p", content: "HTML/CSS/JavaScript: For crafting a responsive and intuitive front-end." },
          { type: "p", content: "Pandas and NLTK: Used in data manipulation and natural language processing respectively" },
          { type: "p", content: "Scikit-Learn: Employed for machine learning algorithms including TF-IDF vectorization and cosine similarity for song recommendations." },
          { type: "p", content: "Spotipy: A lightweight Python library for the Spotify Web API." },
          { type: "h2", content: "Challenges" },
          { type: "p", content: "Data Handling: Managing and processing a large dataset from Spotify was challenging, especially ensuring the speed and efficiency of the system." },
          { type: "p", content: "Machine Learning Implementation: Tuning the machine learning model to improve recommendation accuracy involved significant experimentation and optimization." },
          { type: "p", content: "API Integration: Integrating the Spotify API required understanding its limits and managing authentication and session management efficiently." },
          { type: "h2", content: "Results" },
          { type: "p", content: "The Music Recommender System successfully provides users with song suggestions based on their input. It can dynamically adjust recommendations based on varying user inputs and has shown a high level of accuracy and user satisfaction in preliminary feedback." },
          { type: "h2", content: "Learning Outcomes" },
          { type: "p", content: "Gained deeper insights into the application of natural language processing in real-world projects." },
          { type: "p", content: "Enhanced skills in full-stack web development and learned to handle API integrations more effectively." },
          { type: "p", content: "Improved understanding of machine learning model deployment in web applications, dealing with live data and user interactions." },
          
        ],
      },
      
    ];

    return (
      <div name="portfolio" className="bg-gradient-to-b from-gray-800 to-black w-full text-white overflow-hidden">
        <div className="max-w-screen-lg p-4 mx-auto flex flex-col justify-center w-full">
  <div className="pt-20 pb-8">
    <p className="text-4xl font-bold inline border-b-4 border-gray-500 text-left">Portfolio</p>
  </div>
  
          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-8 px-12 sm:px-0">
          {portfolios.map((project) => (
  <div key={project.id} className="shadow-md shadow-gray-600 rounded-lg overflow-hidden flex flex-col justify-between">
    <div>
      <img src={project.src} alt="" className="rounded-t-md w-full h-auto object-cover" />
      <div className="p-4 text-center">
        <p className="font-semibold text-lg mb-1 text-white">{project.title}</p>
        <p className="text-gray-500 text-sm mb-4">{project.info}</p> {/* Adjust margin as needed */}
      </div>
    </div>
    <div className="flex items-center justify-center p-4">
      <button onClick={() => openModal(project)} className="w-1/2 px-6 py-3 m-2 duration-200 hover:scale-105 bg-green-500 text-white rounded-lg">
        ReadMe
      </button>
      <a href={project.githubUrl} target="_blank" rel="noopener noreferrer" className="inline-block w-1/2 px-6 py-3 m-2 text-center duration-200 hover:scale-105 bg-green-500 text-white rounded-lg">
        Code
      </a>
    </div>
  </div>
))}
          </div>
        </div>
        {selectedProject && (
  <InfoModal
    isOpen={!!selectedProject}
    onClose={closeModal}
    title={selectedProject.title}
    ReadMe={selectedProject.ReadMe} // Ensure this line is correctly passing the ReadMe data
  />
)}
      </div>
    );
  };
