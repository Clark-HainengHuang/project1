# 14813 final project option1 (fifa data)
the demonstrating video with explanation (about 20 min) sharing link is <<https://cmu.box.com/s/7a8qtvzmpcko0ihl2o4q5b2x2o1aapcw>>

the entire demonstrating video (about 1 hour) sharing link is <<https://cmu.box.com/s/qxl08lkih5vvmercgexogxykv9yfej36>>

co_work of

hainengh,\
Haineng Huang
    
kaixins,\
Kaixin Song

## 1 Docker setup
this part has largely referenced to the work of Yiren Zhou <<yirenzho@andrew.cmu.edu>>, and uses the Docker file provided by Yiren Zhou, thanks very much.\
The oringinal work can be visited here: <<https://github.com/ML-Systems-and-Toolchains/course-project-option-1-sample-2>>
### 1.1 start the contaioner (for both windows and linux system)

1.Please ensure that Docker is available (including docker compose).\
2.download the code from github,unzip if needed. For Linux, we suggest to put the folder under `/home/`\
3.open a cmd window (Windows) or a terminal (Linux)\
4.`cd` to downloaded file location\
5.execute `docker compose up -d`,which will install all the necessary dependencies.\
6.after the container starts, get into workspace by `docker exec -it --user root cp /bin/bash`
### 1.2 change user password of PostgreSQL and create schema
1.start the postgresql server by `service postgresql start`\
2.get into postgresql workspace by `sudo -u postgres psql`\
3.execute the following two SQL statements to 1) change the password and 2) create a schema named "fifa":\
`ALTER USER postgres PASSWORD 'bigdata';`\
`CREATE SCHEMA fifa;`\
4.exit the shell by `\q`
### 1.3 open jupyter notebook and execute code for Task 1_2_3
1.start the Jupyter notebook within the container by `jupyter notebook --ip 0.0.0.0 --allow-root`\
2.you should see a URL like: `http://127.0.0.1:8888/?token=574bb631fc2e3c790dcf6c0317f9f3ae674cab80264f1707`\
3.copy the URL (better use the one with `127.0.0.1`) and open it in web broswer\
4.Open <Task_1_2_3.ipynb>, and execute the cells sequentially.
### 1.4 Exit and stop container

1.exiting the Docker container by `Ctrl + D`

2.execute the following to stop the container

```
docker compose down
```
## 2 Task-I:Build and populate necessary tables
### 2.1 Build and populate necessary tables
implement the code cells in the jupyter notebook for the step
### 2.2 setting the constraint for table
(for both windows & linux)\
1.`cd` to docker file location\
2.get into workspace by `docker exec -it --user root cp /bin/bash`\
3.get into postgresql workspace by `sudo -u postgres psql`\
3.setting the constraint for table by following code\
```
ALTER TABLE fifa.fifa
    ADD CONSTRAINT tablename_pkey 
        PRIMARY KEY (id); 
```
### 2.3 write the pyspark dataframe to postgresql table
implement the code cells in the jupyter notebook for the step
### 2.4 (Attachment) Descriptions for features

 'sofifa_id',
     id of players in fifa
     
 'player_url',
 'short_name',
 'long_name',
     name and personal website url
     
 'player_positions',
 'overall',
 'potential',
 'value_eur',
 'wage_eur',
 'age',
 'dob',
 'height_cm',
 'weight_kg',
     basical informations for players 
     
 'club_team_id',
 'club_name',
 'league_name',
 'league_level',
 'club_position',
 'club_jersey_number',
 'club_loaned_from',
 'club_joined',
 'club_contract_valid_until',
 'nationality_id',
 'nationality_name',
 'nation_team_id',
 'nation_position',
 'nation_jersey_number',
     club and nation information
     
 'preferred_foot',
 'weak_foot',
 'skill_moves',
 'international_reputation',
 'work_rate',
 'body_type',
 'real_face',
 'release_clause_eur',
     information for personal 
     
 'player_tags',
     tag on players
     
 'player_traits',
     special traits of players
     
 'pace',
 'shooting',
 'passing',
 'dribbling',
 'defending',
 'physic',
 'attacking_crossing',
 'attacking_finishing',
 'attacking_heading_accuracy',
 'attacking_short_passing',
 'attacking_volleys',
 'skill_dribbling',
 'skill_curve',
 'skill_fk_accuracy',
 'skill_long_passing',
 'skill_ball_control',
 'movement_acceleration',
 'movement_sprint_speed',
 'movement_agility',
 'movement_reactions',
 'movement_balance',
 'power_shot_power',
 'power_jumping',
 'power_stamina',
 'power_strength',
 'power_long_shots',
 'mentality_aggression',
 'mentality_interceptions',
 'mentality_positioning',
 'mentality_vision',
 'mentality_penalties',
 'mentality_composure',
 'defending_marking_awareness',
 'defending_standing_tackle',
 'defending_sliding_tackle',
 'goalkeeping_diving',
 'goalkeeping_handling',
 'goalkeeping_kicking',
 'goalkeeping_positioning',
 'goalkeeping_reflexes',
 'goalkeeping_speed',
     score/grades on performance
     
 'ls',
 'st',
 'rs',
 'lw',
 'lf',
 'cf',
 'rf',
 'rw',
 'lam',
 'cam',
 'ram',
 'lm',
 'lcm',
 'cm',
 'rcm',
 'rm',
 'lwb',
 'ldm',
 'cdm',
 'rdm',
 'rwb',
 'lb',
 'lcb',
 'cb',
 'rcb',
 'rb',
 'gk',
     score/grades for different positions on the playground
     
 'player_face_url',
 'club_logo_url',
 'club_flag_url',
 'nation_logo_url',
 'nation_flag_url',
     picture url for players and their club, nation
     
 'year'
     the year of csv files this row of data came from
## 3 Task-II: Conduct analytics on your dataset
### 3.1 read the data from postgresql 
implement the code cells in the jupyter notebook for the step
### 3.2 Functions for analyze
implement the code cells in the jupyter notebook for the step
## 4 Task-III: Machine Learning Modeling
### 4.1 read the data from postgresql 
implement the code cells in the jupyter notebook for the step
### 4.2 Data Engineering
#### Drop columns 
the following descriptions referenced to the work of Yiren Zhou <<yirenzho@andrew.cmu.edu>>, thanks very much.\
https://github.com/ML-Systems-and-Toolchains/course-project-option-1-sample-2

I have decided to drop the following columns as they don't seem relevant to the overall value of a player:

player_url (not relevant)\
short_name (not relevant)\
long_name (not relevant) \
club_name (too many values, could be useless) \
club_loaned_from (too few values) \
club_joined(not relevant)\
dob(not relevant)\
release_clause_eur (too few values) \
player_positions (cannot be converted into numerics easily) \
goalkeeping_speed (too few values) \
player_traits (too random) \
player_tags (there are missing entries for certain players)\
real_face (not relevant) \
mentality_composure (too few values) \
player_face_url (not relevant) \
club_logo_url (not relevant) \
club_flag_url (not relevant) \
nation_logo_url (not relevant)\
nation_flag_url (not relevant)\
ls (not relevant)\
.\
.\
.\
gk(not relevant)
### 4.3 model on spark

I choosed LinearRegression and RandomForestRegressor.

since it's a regression task, I choose MSE(mean squared error) as the target.

the dataset is devided to train(80%),test(20%)
#### 4.3.1 LinearRegression

"regParam" is regularization parameter, which influence the regularization process and thus contributes to model performance. The default value is '0.0'

"maxIter" is number of iterations, which determines the training times and convergence of the model. The defualt value is '100'
#### 4.3.2 RandomForestRegressor
"maxDepth" means the maxmum depth of trees in the RF model, the default value is 4

"numTrees" means the number of trees in the RF model,
### 4.4 Model on Tensorflow
I built Basic regression Model (only 1 layer, which is like Linear Regression) and Deep Nueral Network(more than 3 layers)

since it's a regression task, I choose MSE(mean squared error) as the target, which is also the loss I used.

the dataset is devided to train(80%),validate(10%),test(10%)
#### 4.4.1 Basic regression 
the Basic regression model only has 1 layer, and choosed the loss of 'mean_squared_error'

the hyperpamameter is:\
learningrate = [0.01,0.1]\
epochs = [10,100]
#### 4.4.2 Deep Nueral Network
the Deep Nueral Network model has more than 3 layers, and choosed the loss of 'mean_squared_error', the activation function of 'relu',epochs = 100

the hyperpamameter is:\
WIDTH = [10,15,20]\
DEPTH = [3,5,7]
## 5 Task-4 Deploy your code to the Cloud
there is two ways to deploy the code to the Cloud\
one is using Google Cluster or Amazon AWS and so on, which can run the docker container\
the other is using Google Colab (the Colab will clean all the data if closed). The following part is run on Google Colab\
in this part I used Colab\
1.up load and open `Task-4.ipynb` on Colab
### 5.1 install pyspark,postgresql and set the password for postgresql
implement the code cells in the jupyter notebook for the step
### 5.2 copy the jar file and data to drive location
1.upload `postgresql-42.5.0.jar` to the location of "Task-4.ipynb". In my part (default setting) is: `/content/drive/MyDrive/Colab Notebooks/`\
2.upload `data` folder to the same location with Task-4. sometimes it will fail to upload the entire folder, we can new build a folder named `data` and upload all the csv files to it.
### 5.3 task-2 part on cloud
just as task-2 part, implement the code cells in the jupyter notebook for the step
### 5.4 task-3 part on cloud
just as task-3 part, implement the code cells in the jupyter notebook for the step
### 5.5 task-4 part on cloud
just as task-4 part, implement the code cells in the jupyter notebook for the step\
some tiny changes are made to save training time (in the Deep Nueral Network part, I set the width and depth to [10,15] and [3,5], epochs = 30)
