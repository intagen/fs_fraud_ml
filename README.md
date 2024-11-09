# fs_fraud_ml
Banking and financial services fraud detection with machine learning.

```
$ python main.py
Path to dataset files: /Users/pawel/.cache/kagglehub/datasets/goyaladi/fraud-detection-dataset/versions/3
Transactions dataframe columns: ['TransactionID', 'Amount', 'CustomerID', 'Timestamp', 'MerchantID', 'FraudIndicator', 'SuspiciousFlag']
Merchant data dataframe columns: ['MerchantID', 'MerchantName', 'Location']
Transaction category labels dataframe columns: ['TransactionID', 'Category']
Non-numeric columns: ['MerchantName', 'Location', 'Name', 'Address', 'LastLogin']
Resampled training set shape: (1492, 21)

Training XGBoost Classifier...
[I 2024-11-09 15:35:39,723] A new study created in memory with name: no-name-2a38bcc0-841f-420b-9c61-cdb4368ede7d
[I 2024-11-09 15:35:41,706] Trial 0 finished with value: 0.736730752450781 and parameters: {'max_depth': 3, 'learning_rate': 0.009003385115271476, 'n_estimators': 240, 'gamma': 0.10516015459252753, 'min_child_weight': 7, 'subsample': 0.7771038617913579, 'colsample_bytree': 0.9854630159721657, 'reg_alpha': 0.7901048598067999, 'reg_lambda': 0.02398574931176947, 'scale_pos_weight': 4.27170517127496}. Best is trial 0 with value: 0.736730752450781.
[I 2024-11-09 15:35:42,423] Trial 1 finished with value: 0.9445081033756028 and parameters: {'max_depth': 10, 'learning_rate': 0.16112597130416012, 'n_estimators': 112, 'gamma': 0.4889257070279707, 'min_child_weight': 9, 'subsample': 0.6382736593335192, 'colsample_bytree': 0.8541397918118081, 'reg_alpha': 0.002833066912606011, 'reg_lambda': 0.029964084485340527, 'scale_pos_weight': 2.6546672083678593}. Best is trial 1 with value: 0.9445081033756028.
[I 2024-11-09 15:35:43,165] Trial 2 finished with value: 0.7343817593379333 and parameters: {'max_depth': 4, 'learning_rate': 0.008644800534692942, 'n_estimators': 151, 'gamma': 0.41590606869232133, 'min_child_weight': 1, 'subsample': 0.7875474625506907, 'colsample_bytree': 0.8910532432055773, 'reg_alpha': 0.07844120346363509, 'reg_lambda': 0.23392145086504884, 'scale_pos_weight': 4.409776570814753}. Best is trial 1 with value: 0.9445081033756028.
[I 2024-11-09 15:35:43,894] Trial 3 finished with value: 0.7420334639774385 and parameters: {'max_depth': 5, 'learning_rate': 0.00552752094661434, 'n_estimators': 222, 'gamma': 0.14885119816848563, 'min_child_weight': 6, 'subsample': 0.6279146706066928, 'colsample_bytree': 0.6483941985617381, 'reg_alpha': 0.012755226828788852, 'reg_lambda': 0.3978636619354248, 'scale_pos_weight': 3.4004349623866847}. Best is trial 1 with value: 0.9445081033756028.
[I 2024-11-09 15:35:44,662] Trial 4 finished with value: 0.9271259122895191 and parameters: {'max_depth': 5, 'learning_rate': 0.02729243446658895, 'n_estimators': 187, 'gamma': 0.26822558227442206, 'min_child_weight': 9, 'subsample': 0.9884173202864579, 'colsample_bytree': 0.8887868847774798, 'reg_alpha': 0.0189226232603769, 'reg_lambda': 0.002913300283252142, 'scale_pos_weight': 1.7682380849463657}. Best is trial 1 with value: 0.9445081033756028.
[I 2024-11-09 15:35:44,726] Trial 5 finished with value: 0.9456986461923956 and parameters: {'max_depth': 4, 'learning_rate': 0.08103539600447977, 'n_estimators': 133, 'gamma': 0.1327030477000487, 'min_child_weight': 4, 'subsample': 0.6909795998115175, 'colsample_bytree': 0.6772755568323824, 'reg_alpha': 0.16832627300647904, 'reg_lambda': 0.2866650792492319, 'scale_pos_weight': 1.4490947530665421}. Best is trial 5 with value: 0.9456986461923956.
[I 2024-11-09 15:35:44,851] Trial 6 finished with value: 0.9492563114274623 and parameters: {'max_depth': 10, 'learning_rate': 0.049306703173294164, 'n_estimators': 220, 'gamma': 0.0803347742483243, 'min_child_weight': 6, 'subsample': 0.7806745083205823, 'colsample_bytree': 0.7241564036935578, 'reg_alpha': 0.03938790443115031, 'reg_lambda': 0.12414390813447274, 'scale_pos_weight': 3.1794388786351875}. Best is trial 6 with value: 0.9492563114274623.
[I 2024-11-09 15:35:44,966] Trial 7 finished with value: 0.8650711365125568 and parameters: {'max_depth': 5, 'learning_rate': 0.013901731324814485, 'n_estimators': 242, 'gamma': 0.1579095889100992, 'min_child_weight': 8, 'subsample': 0.9754013388473438, 'colsample_bytree': 0.7309371494098444, 'reg_alpha': 0.02109834410376582, 'reg_lambda': 0.6298337391820513, 'scale_pos_weight': 3.5195528953078994}. Best is trial 6 with value: 0.9492563114274623.
[I 2024-11-09 15:35:45,042] Trial 8 finished with value: 0.6930602261588176 and parameters: {'max_depth': 4, 'learning_rate': 0.005053710900720599, 'n_estimators': 128, 'gamma': 0.3647335246180638, 'min_child_weight': 4, 'subsample': 0.7601966596396089, 'colsample_bytree': 0.9448372562747924, 'reg_alpha': 0.21694674183694054, 'reg_lambda': 0.039318147959225995, 'scale_pos_weight': 3.6409897523218047}. Best is trial 6 with value: 0.9492563114274623.
[I 2024-11-09 15:35:45,129] Trial 9 finished with value: 0.8847425146176192 and parameters: {'max_depth': 5, 'learning_rate': 0.027809761592765124, 'n_estimators': 110, 'gamma': 0.3077710444062108, 'min_child_weight': 1, 'subsample': 0.6202938603496447, 'colsample_bytree': 0.7120683544977631, 'reg_alpha': 0.015043959315566078, 'reg_lambda': 0.021690887256863106, 'scale_pos_weight': 3.1541855806406405}. Best is trial 6 with value: 0.9492563114274623.
[I 2024-11-09 15:35:45,285] Trial 10 finished with value: 0.9577705232649144 and parameters: {'max_depth': 10, 'learning_rate': 0.07298216279010285, 'n_estimators': 297, 'gamma': 0.004335535924655909, 'min_child_weight': 4, 'subsample': 0.8880670828421474, 'colsample_bytree': 0.7803374712974989, 'reg_alpha': 0.001388964327667063, 'reg_lambda': 0.0022187802993225807, 'scale_pos_weight': 2.3723026754470506}. Best is trial 10 with value: 0.9577705232649144.
[I 2024-11-09 15:35:45,451] Trial 11 finished with value: 0.954755316967414 and parameters: {'max_depth': 10, 'learning_rate': 0.063322258096662, 'n_estimators': 296, 'gamma': 0.0001478185066378368, 'min_child_weight': 4, 'subsample': 0.8895146559294483, 'colsample_bytree': 0.7827206890437932, 'reg_alpha': 0.0021298884451218833, 'reg_lambda': 0.0015009255626881669, 'scale_pos_weight': 2.321204721885405}. Best is trial 10 with value: 0.9577705232649144.
[I 2024-11-09 15:35:45,593] Trial 12 finished with value: 0.9541271660691303 and parameters: {'max_depth': 8, 'learning_rate': 0.09448878972200837, 'n_estimators': 292, 'gamma': 0.009179612966328975, 'min_child_weight': 3, 'subsample': 0.8984129472289947, 'colsample_bytree': 0.8083437957528244, 'reg_alpha': 0.0010366754409833047, 'reg_lambda': 0.00106864467705231, 'scale_pos_weight': 2.1946210650698363}. Best is trial 10 with value: 0.9577705232649144.
[I 2024-11-09 15:35:45,780] Trial 13 finished with value: 0.9578916962025358 and parameters: {'max_depth': 8, 'learning_rate': 0.051946138317218844, 'n_estimators': 295, 'gamma': 0.0024363120891360513, 'min_child_weight': 3, 'subsample': 0.8855739104354456, 'colsample_bytree': 0.7906423044868727, 'reg_alpha': 0.0036669120443453573, 'reg_lambda': 0.004516380858246129, 'scale_pos_weight': 2.1119957269006053}. Best is trial 13 with value: 0.9578916962025358.
[I 2024-11-09 15:35:45,893] Trial 14 finished with value: 0.9613607397752931 and parameters: {'max_depth': 8, 'learning_rate': 0.1416119411096815, 'n_estimators': 273, 'gamma': 0.21530821995028848, 'min_child_weight': 2, 'subsample': 0.8708667116753077, 'colsample_bytree': 0.7884355755740879, 'reg_alpha': 0.004908925712705289, 'reg_lambda': 0.004251124212313836, 'scale_pos_weight': 1.1752082025479238}. Best is trial 14 with value: 0.9613607397752931.
[I 2024-11-09 15:35:45,985] Trial 15 finished with value: 0.9632719297266364 and parameters: {'max_depth': 8, 'learning_rate': 0.19511622876791504, 'n_estimators': 259, 'gamma': 0.21162419261611745, 'min_child_weight': 2, 'subsample': 0.8478127418482443, 'colsample_bytree': 0.6023997186416822, 'reg_alpha': 0.005330319648703824, 'reg_lambda': 0.007372075345595139, 'scale_pos_weight': 1.4467640192018671}. Best is trial 15 with value: 0.9632719297266364.
[I 2024-11-09 15:35:46,718] Trial 16 finished with value: 0.9622890276685975 and parameters: {'max_depth': 8, 'learning_rate': 0.1939251636479552, 'n_estimators': 260, 'gamma': 0.21636840990912973, 'min_child_weight': 2, 'subsample': 0.8516345412297254, 'colsample_bytree': 0.6038354456468423, 'reg_alpha': 0.006388036132647687, 'reg_lambda': 0.006896023984554661, 'scale_pos_weight': 1.182375688270997}. Best is trial 15 with value: 0.9632719297266364.
[I 2024-11-09 15:35:47,418] Trial 17 finished with value: 0.9664475348306868 and parameters: {'max_depth': 7, 'learning_rate': 0.18001407188815996, 'n_estimators': 261, 'gamma': 0.2146468074668317, 'min_child_weight': 2, 'subsample': 0.8351396379515685, 'colsample_bytree': 0.614354218458588, 'reg_alpha': 0.008055129685774197, 'reg_lambda': 0.011869875730772923, 'scale_pos_weight': 1.091892740211627}. Best is trial 17 with value: 0.9664475348306868.
[I 2024-11-09 15:35:47,512] Trial 18 finished with value: 0.9601459654487975 and parameters: {'max_depth': 7, 'learning_rate': 0.12160073222126382, 'n_estimators': 184, 'gamma': 0.30353104504291617, 'min_child_weight': 2, 'subsample': 0.7348711278790608, 'colsample_bytree': 0.6038263294213114, 'reg_alpha': 0.05148026621663717, 'reg_lambda': 0.010589361324672345, 'scale_pos_weight': 1.6894085544671151}. Best is trial 17 with value: 0.9664475348306868.
[I 2024-11-09 15:35:47,618] Trial 19 finished with value: 0.955060200437388 and parameters: {'max_depth': 7, 'learning_rate': 0.10233175267673704, 'n_estimators': 265, 'gamma': 0.20661408191439942, 'min_child_weight': 5, 'subsample': 0.8299509804394091, 'colsample_bytree': 0.6478079402093379, 'reg_alpha': 0.008871087125744427, 'reg_lambda': 0.011554605141115502, 'scale_pos_weight': 1.1306345866562577}. Best is trial 17 with value: 0.9664475348306868.

xgboost best hyperparameters:
{'max_depth': 7, 'learning_rate': 0.18001407188815996, 'n_estimators': 261, 'gamma': 0.2146468074668317, 'min_child_weight': 2, 'subsample': 0.8351396379515685, 'colsample_bytree': 0.614354218458588, 'reg_alpha': 0.008055129685774197, 'reg_lambda': 0.011869875730772923, 'scale_pos_weight': 1.091892740211627}

XGBoost - Best threshold based on f1 score: 0.0159

XGBoost - Classification report:
              precision    recall  f1-score   support

           0     0.9643    0.5775    0.7224       187
           1     0.1023    0.6923    0.1782        13

    accuracy                         0.5850       200
   macro avg     0.5333    0.6349    0.4503       200
weighted avg     0.9083    0.5850    0.6870       200

XGBoost - Confusion matrix:
[[108  79]
 [  4   9]]
XGBoost - ROC AUC score: 0.6018
XGBoost - Average precision score: 0.0926

Training Random Forest Classifier...
[I 2024-11-09 15:36:01,517] A new study created in memory with name: no-name-922960f1-718b-438d-b847-93321d94c2d5
[I 2024-11-09 15:36:01,839] Trial 0 finished with value: 0.9656213933849402 and parameters: {'n_estimators': 115, 'max_depth': 20, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample'}. Best is trial 0 with value: 0.9656213933849402.
[I 2024-11-09 15:36:02,083] Trial 1 finished with value: 0.9615689680748055 and parameters: {'n_estimators': 101, 'max_depth': 14, 'min_samples_split': 4, 'min_samples_leaf': 3, 'max_features': 'log2', 'class_weight': 'balanced_subsample'}. Best is trial 0 with value: 0.9656213933849402.
[I 2024-11-09 15:36:02,510] Trial 2 finished with value: 0.9560054794994013 and parameters: {'n_estimators': 217, 'max_depth': 9, 'min_samples_split': 8, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample'}. Best is trial 0 with value: 0.9656213933849402.
[I 2024-11-09 15:36:03,049] Trial 3 finished with value: 0.9561686468089746 and parameters: {'n_estimators': 227, 'max_depth': 9, 'min_samples_split': 2, 'min_samples_leaf': 3, 'max_features': None, 'class_weight': None}. Best is trial 0 with value: 0.9656213933849402.
[I 2024-11-09 15:36:03,280] Trial 4 finished with value: 0.9724189214539143 and parameters: {'n_estimators': 181, 'max_depth': 18, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:03,862] Trial 5 finished with value: 0.9497751060448577 and parameters: {'n_estimators': 262, 'max_depth': 8, 'min_samples_split': 3, 'min_samples_leaf': 3, 'max_features': None, 'class_weight': 'balanced'}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:04,136] Trial 6 finished with value: 0.9472354062715508 and parameters: {'n_estimators': 119, 'max_depth': 8, 'min_samples_split': 4, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample'}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:04,519] Trial 7 finished with value: 0.971052351798421 and parameters: {'n_estimators': 219, 'max_depth': 16, 'min_samples_split': 3, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample'}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:04,794] Trial 8 finished with value: 0.9622041336579827 and parameters: {'n_estimators': 151, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample'}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:05,013] Trial 9 finished with value: 0.9501111577574037 and parameters: {'n_estimators': 122, 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample'}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:05,239] Trial 10 finished with value: 0.9587565997534062 and parameters: {'n_estimators': 170, 'max_depth': 20, 'min_samples_split': 7, 'min_samples_leaf': 5, 'max_features': 'log2', 'class_weight': None}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:05,490] Trial 11 finished with value: 0.9670395725204214 and parameters: {'n_estimators': 190, 'max_depth': 17, 'min_samples_split': 6, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': None}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:05,880] Trial 12 finished with value: 0.9717363372948705 and parameters: {'n_estimators': 272, 'max_depth': 17, 'min_samples_split': 3, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced'}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:06,249] Trial 13 finished with value: 0.9690433667628339 and parameters: {'n_estimators': 299, 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'log2', 'class_weight': 'balanced'}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:07,642] Trial 14 finished with value: 0.9644893636829122 and parameters: {'n_estimators': 269, 'max_depth': 18, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': None, 'class_weight': 'balanced'}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:07,958] Trial 15 finished with value: 0.9697627327595262 and parameters: {'n_estimators': 246, 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': None}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:08,198] Trial 16 finished with value: 0.9620580083378836 and parameters: {'n_estimators': 185, 'max_depth': 18, 'min_samples_split': 3, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'class_weight': 'balanced'}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:08,416] Trial 17 finished with value: 0.9704930291134134 and parameters: {'n_estimators': 152, 'max_depth': 13, 'min_samples_split': 6, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:09,218] Trial 18 finished with value: 0.9631592167425556 and parameters: {'n_estimators': 298, 'max_depth': 18, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': None, 'class_weight': 'balanced'}. Best is trial 4 with value: 0.9724189214539143.
[I 2024-11-09 15:36:09,510] Trial 19 finished with value: 0.9203721359046186 and parameters: {'n_estimators': 244, 'max_depth': 6, 'min_samples_split': 7, 'min_samples_leaf': 4, 'max_features': 'log2', 'class_weight': None}. Best is trial 4 with value: 0.9724189214539143.

random forest best hyperparameters:
{'n_estimators': 181, 'max_depth': 18, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None}

Random Forest - Best threshold based on f1 score: 0.2585

Random Forest - Classification report:
              precision    recall  f1-score   support

           0     0.9551    0.7968    0.8688       187
           1     0.1364    0.4615    0.2105        13

    accuracy                         0.7750       200
   macro avg     0.5457    0.6292    0.5397       200
weighted avg     0.9019    0.7750    0.8260       200

Random Forest - Confusion matrix:
[[149  38]
 [  7   6]]
Random Forest - ROC AUC score: 0.6598
Random Forest - Average precision score: 0.1036

Training LightGBM Classifier...
[I 2024-11-09 15:36:18,997] A new study created in memory with name: no-name-e912735a-57e7-48c3-a9ed-6331aa0c2699
[I 2024-11-09 15:36:20,003] Trial 0 finished with value: 0.8768343995487186 and parameters: {'num_leaves': 84, 'learning_rate': 0.011325457666293845, 'n_estimators': 227, 'min_child_samples': 45, 'max_depth': 3, 'subsample': 0.9321005428125335, 'colsample_bytree': 0.927802835101424, 'reg_alpha': 0.4073166474497316, 'reg_lambda': 0.995926170136965, 'class_weight': 'balanced'}. Best is trial 0 with value: 0.8768343995487186.
[I 2024-11-09 15:36:20,610] Trial 1 finished with value: 0.9531258800029242 and parameters: {'num_leaves': 81, 'learning_rate': 0.08692737440667951, 'n_estimators': 282, 'min_child_samples': 50, 'max_depth': 4, 'subsample': 0.6142991700621078, 'colsample_bytree': 0.9503439741625477, 'reg_alpha': 0.0031122357887954275, 'reg_lambda': 0.9458315460088443, 'class_weight': 'balanced'}. Best is trial 1 with value: 0.9531258800029242.
[I 2024-11-09 15:36:20,953] Trial 2 finished with value: 0.9356631006909119 and parameters: {'num_leaves': 41, 'learning_rate': 0.0961398537326651, 'n_estimators': 130, 'min_child_samples': 30, 'max_depth': 3, 'subsample': 0.6063281553504073, 'colsample_bytree': 0.83513736241028, 'reg_alpha': 0.0019016088082414186, 'reg_lambda': 0.006973748190574985, 'class_weight': 'balanced'}. Best is trial 1 with value: 0.9531258800029242.
[I 2024-11-09 15:36:21,339] Trial 3 finished with value: 0.9319010492272879 and parameters: {'num_leaves': 41, 'learning_rate': 0.05642525519348337, 'n_estimators': 180, 'min_child_samples': 40, 'max_depth': 3, 'subsample': 0.7199708566282256, 'colsample_bytree': 0.6463107928390929, 'reg_alpha': 0.08842905248674388, 'reg_lambda': 0.002121852328508583, 'class_weight': None}. Best is trial 1 with value: 0.9531258800029242.
[I 2024-11-09 15:36:21,945] Trial 4 finished with value: 0.9481911438158853 and parameters: {'num_leaves': 95, 'learning_rate': 0.05671674003268413, 'n_estimators': 270, 'min_child_samples': 47, 'max_depth': 4, 'subsample': 0.7969301406909576, 'colsample_bytree': 0.8227480245722573, 'reg_alpha': 0.018939403806306977, 'reg_lambda': 0.8758329149392452, 'class_weight': None}. Best is trial 1 with value: 0.9531258800029242.
[I 2024-11-09 15:36:23,067] Trial 5 finished with value: 0.9390480116195098 and parameters: {'num_leaves': 95, 'learning_rate': 0.010274531646966746, 'n_estimators': 247, 'min_child_samples': 21, 'max_depth': 6, 'subsample': 0.8605354824018172, 'colsample_bytree': 0.703415050171414, 'reg_alpha': 0.0013245687876313576, 'reg_lambda': 0.011190912730250278, 'class_weight': 'balanced'}. Best is trial 1 with value: 0.9531258800029242.
[I 2024-11-09 15:36:24,409] Trial 6 finished with value: 0.9590368970791516 and parameters: {'num_leaves': 25, 'learning_rate': 0.02116779520091349, 'n_estimators': 168, 'min_child_samples': 22, 'max_depth': 13, 'subsample': 0.929811293594863, 'colsample_bytree': 0.782737907553941, 'reg_alpha': 0.011169367001544743, 'reg_lambda': 0.12645935978489461, 'class_weight': None}. Best is trial 6 with value: 0.9590368970791516.
[I 2024-11-09 15:36:25,310] Trial 7 finished with value: 0.9658597982751423 and parameters: {'num_leaves': 99, 'learning_rate': 0.18120335727486125, 'n_estimators': 233, 'min_child_samples': 38, 'max_depth': 6, 'subsample': 0.8922857437071703, 'colsample_bytree': 0.6479357767648483, 'reg_alpha': 0.11926414785029872, 'reg_lambda': 0.8106476792922931, 'class_weight': None}. Best is trial 7 with value: 0.9658597982751423.
[I 2024-11-09 15:36:26,080] Trial 8 finished with value: 0.907079470042433 and parameters: {'num_leaves': 26, 'learning_rate': 0.010583363409018732, 'n_estimators': 162, 'min_child_samples': 45, 'max_depth': 9, 'subsample': 0.6687823681196275, 'colsample_bytree': 0.8569986573249555, 'reg_alpha': 0.27226384188903896, 'reg_lambda': 0.048221399728615895, 'class_weight': None}. Best is trial 7 with value: 0.9658597982751423.
[I 2024-11-09 15:36:26,554] Trial 9 finished with value: 0.9644275930804924 and parameters: {'num_leaves': 45, 'learning_rate': 0.1257172043037641, 'n_estimators': 111, 'min_child_samples': 49, 'max_depth': 14, 'subsample': 0.8356864013112424, 'colsample_bytree': 0.6509616220925305, 'reg_alpha': 0.003058536763053957, 'reg_lambda': 0.15602699540387494, 'class_weight': 'balanced'}. Best is trial 7 with value: 0.9658597982751423.
[I 2024-11-09 15:36:27,405] Trial 10 finished with value: 0.9685282953620646 and parameters: {'num_leaves': 66, 'learning_rate': 0.1980758242826311, 'n_estimators': 218, 'min_child_samples': 33, 'max_depth': 8, 'subsample': 0.9778108042674867, 'colsample_bytree': 0.7359066176573372, 'reg_alpha': 0.07900373673552777, 'reg_lambda': 0.1641150532279246, 'class_weight': None}. Best is trial 10 with value: 0.9685282953620646.
[I 2024-11-09 15:36:28,433] Trial 11 finished with value: 0.9652261701493347 and parameters: {'num_leaves': 63, 'learning_rate': 0.1662332854054672, 'n_estimators': 213, 'min_child_samples': 34, 'max_depth': 8, 'subsample': 0.9911037800267373, 'colsample_bytree': 0.7198487026277616, 'reg_alpha': 0.07074281018379466, 'reg_lambda': 0.23081593389247604, 'class_weight': None}. Best is trial 10 with value: 0.9685282953620646.
[I 2024-11-09 15:36:29,426] Trial 12 finished with value: 0.9684113069729458 and parameters: {'num_leaves': 74, 'learning_rate': 0.18262943578413976, 'n_estimators': 234, 'min_child_samples': 30, 'max_depth': 11, 'subsample': 0.9834226750332167, 'colsample_bytree': 0.6005399035627489, 'reg_alpha': 0.08998149855805689, 'reg_lambda': 0.2932657977398526, 'class_weight': None}. Best is trial 10 with value: 0.9685282953620646.
[I 2024-11-09 15:36:30,002] Trial 13 finished with value: 0.9643008647325485 and parameters: {'num_leaves': 65, 'learning_rate': 0.19747480173065304, 'n_estimators': 300, 'min_child_samples': 29, 'max_depth': 11, 'subsample': 0.9933857719927028, 'colsample_bytree': 0.6069784745339254, 'reg_alpha': 0.7823638833578183, 'reg_lambda': 0.0556875258334328, 'class_weight': None}. Best is trial 10 with value: 0.9685282953620646.
[I 2024-11-09 15:36:31,260] Trial 14 finished with value: 0.9650534222451742 and parameters: {'num_leaves': 75, 'learning_rate': 0.03574676651453329, 'n_estimators': 198, 'min_child_samples': 28, 'max_depth': 11, 'subsample': 0.7781294782588446, 'colsample_bytree': 0.7407620012975089, 'reg_alpha': 0.03550755169230541, 'reg_lambda': 0.29457040313201605, 'class_weight': None}. Best is trial 10 with value: 0.9685282953620646.
[I 2024-11-09 15:36:32,299] Trial 15 finished with value: 0.9690656968272044 and parameters: {'num_leaves': 52, 'learning_rate': 0.10935583339334112, 'n_estimators': 257, 'min_child_samples': 34, 'max_depth': 11, 'subsample': 0.9433562595376999, 'colsample_bytree': 0.9992107032331536, 'reg_alpha': 0.18886681601071884, 'reg_lambda': 0.01727219418899528, 'class_weight': None}. Best is trial 15 with value: 0.9690656968272044.
[I 2024-11-09 15:36:33,270] Trial 16 finished with value: 0.9643745505946689 and parameters: {'num_leaves': 55, 'learning_rate': 0.10914677104799, 'n_estimators': 258, 'min_child_samples': 38, 'max_depth': 7, 'subsample': 0.9218738698359903, 'colsample_bytree': 0.98532830884974, 'reg_alpha': 0.23155270733788821, 'reg_lambda': 0.017141467980836145, 'class_weight': None}. Best is trial 15 with value: 0.9690656968272044.
[I 2024-11-09 15:36:34,266] Trial 17 finished with value: 0.9632144310600608 and parameters: {'num_leaves': 52, 'learning_rate': 0.07635039488975441, 'n_estimators': 204, 'min_child_samples': 33, 'max_depth': 10, 'subsample': 0.949808812606384, 'colsample_bytree': 0.8871271938963234, 'reg_alpha': 0.9923062463975582, 'reg_lambda': 0.0016528698740662805, 'class_weight': None}. Best is trial 15 with value: 0.9690656968272044.
[I 2024-11-09 15:36:36,369] Trial 18 finished with value: 0.9691355108527001 and parameters: {'num_leaves': 34, 'learning_rate': 0.03408130760193008, 'n_estimators': 290, 'min_child_samples': 25, 'max_depth': 13, 'subsample': 0.8728068368889936, 'colsample_bytree': 0.7756775525314898, 'reg_alpha': 0.01024289929003456, 'reg_lambda': 0.00417547824016551, 'class_weight': None}. Best is trial 18 with value: 0.9691355108527001.
[I 2024-11-09 15:36:38,450] Trial 19 finished with value: 0.9698367148190229 and parameters: {'num_leaves': 33, 'learning_rate': 0.03355077858235393, 'n_estimators': 280, 'min_child_samples': 25, 'max_depth': 13, 'subsample': 0.8638916014521634, 'colsample_bytree': 0.7932280209804035, 'reg_alpha': 0.00948625547067861, 'reg_lambda': 0.0039031696563207124, 'class_weight': None}. Best is trial 19 with value: 0.9698367148190229.

lightgbm best hyperparameters:
{'num_leaves': 33, 'learning_rate': 0.03355077858235393, 'n_estimators': 280, 'min_child_samples': 25, 'max_depth': 13, 'subsample': 0.8638916014521634, 'colsample_bytree': 0.7932280209804035, 'reg_alpha': 0.00948625547067861, 'reg_lambda': 0.0039031696563207124, 'class_weight': None}

LightGBM - Best threshold based on f1 score: 0.0241

LightGBM - Classification report:
              precision    recall  f1-score   support

           0     0.9568    0.7112    0.8160       187
           1     0.1148    0.5385    0.1892        13

    accuracy                         0.7000       200
   macro avg     0.5358    0.6248    0.5026       200
weighted avg     0.9021    0.7000    0.7752       200

LightGBM - Confusion matrix:
[[133  54]
 [  6   7]]
LightGBM - ROC AUC score: 0.6331
LightGBM - Average precision score: 0.0968

Training CatBoost Classifier...
[I 2024-11-09 15:36:48,115] A new study created in memory with name: no-name-64798708-eb3a-4d7d-8b77-e94fe05172a2
[I 2024-11-09 15:36:48,405] Trial 0 finished with value: 0.9492890800324291 and parameters: {'depth': 5, 'learning_rate': 0.06960878172580316, 'n_estimators': 190, 'l2_leaf_reg': 0.006001719589248504, 'border_count': 161, 'scale_pos_weight': 4.6619038709160225}. Best is trial 0 with value: 0.9492890800324291.
[I 2024-11-09 15:36:48,681] Trial 1 finished with value: 0.8487099517623639 and parameters: {'depth': 6, 'learning_rate': 0.005924965359009431, 'n_estimators': 117, 'l2_leaf_reg': 0.003452486785388052, 'border_count': 236, 'scale_pos_weight': 2.69948085643787}. Best is trial 0 with value: 0.9492890800324291.
[I 2024-11-09 15:36:49,078] Trial 2 finished with value: 0.9593699717820385 and parameters: {'depth': 8, 'learning_rate': 0.17936682439115498, 'n_estimators': 111, 'l2_leaf_reg': 0.025731471765138517, 'border_count': 199, 'scale_pos_weight': 3.961057277614097}. Best is trial 2 with value: 0.9593699717820385.
[I 2024-11-09 15:36:49,287] Trial 3 finished with value: 0.9486777851026521 and parameters: {'depth': 4, 'learning_rate': 0.04977612036352135, 'n_estimators': 249, 'l2_leaf_reg': 0.4928805988967099, 'border_count': 34, 'scale_pos_weight': 2.3363578814675594}. Best is trial 2 with value: 0.9593699717820385.
[I 2024-11-09 15:36:49,518] Trial 4 finished with value: 0.9532689239624407 and parameters: {'depth': 6, 'learning_rate': 0.11161534062616588, 'n_estimators': 127, 'l2_leaf_reg': 0.13492373011074368, 'border_count': 164, 'scale_pos_weight': 4.745320121017125}. Best is trial 2 with value: 0.9593699717820385.
[I 2024-11-09 15:36:50,053] Trial 5 finished with value: 0.9737550551165396 and parameters: {'depth': 8, 'learning_rate': 0.16850507858444902, 'n_estimators': 169, 'l2_leaf_reg': 0.0011316422148620244, 'border_count': 180, 'scale_pos_weight': 1.5331365047355425}. Best is trial 5 with value: 0.9737550551165396.
[I 2024-11-09 15:36:50,668] Trial 6 finished with value: 0.8806540934987707 and parameters: {'depth': 8, 'learning_rate': 0.00876416059771583, 'n_estimators': 163, 'l2_leaf_reg': 0.05763411191565053, 'border_count': 254, 'scale_pos_weight': 4.3211851651089095}. Best is trial 5 with value: 0.9737550551165396.
[I 2024-11-09 15:36:50,889] Trial 7 finished with value: 0.7554444500817148 and parameters: {'depth': 4, 'learning_rate': 0.01006708815804523, 'n_estimators': 234, 'l2_leaf_reg': 0.06497541998323261, 'border_count': 204, 'scale_pos_weight': 4.384689494654712}. Best is trial 5 with value: 0.9737550551165396.
[I 2024-11-09 15:36:50,991] Trial 8 finished with value: 0.8724065959025978 and parameters: {'depth': 4, 'learning_rate': 0.03163880455593259, 'n_estimators': 100, 'l2_leaf_reg': 0.015974576664932304, 'border_count': 189, 'scale_pos_weight': 2.0046714096957365}. Best is trial 5 with value: 0.9737550551165396.
[I 2024-11-09 15:36:51,438] Trial 9 finished with value: 0.9828631086581128 and parameters: {'depth': 7, 'learning_rate': 0.08934767041781333, 'n_estimators': 295, 'l2_leaf_reg': 0.03613236262466834, 'border_count': 105, 'scale_pos_weight': 2.637135809006868}. Best is trial 9 with value: 0.9828631086581128.
[I 2024-11-09 15:36:52,900] Trial 10 finished with value: 0.966308419050952 and parameters: {'depth': 10, 'learning_rate': 0.02745083265955229, 'n_estimators': 300, 'l2_leaf_reg': 0.992243909862797, 'border_count': 98, 'scale_pos_weight': 3.244230705439704}. Best is trial 9 with value: 0.9828631086581128.
[I 2024-11-09 15:36:53,568] Trial 11 finished with value: 0.9729954751131222 and parameters: {'depth': 8, 'learning_rate': 0.19858561593788665, 'n_estimators': 300, 'l2_leaf_reg': 0.001104215821333564, 'border_count': 110, 'scale_pos_weight': 1.2183119649752152}. Best is trial 9 with value: 0.9828631086581128.
[I 2024-11-09 15:36:54,559] Trial 12 finished with value: 0.975602600573132 and parameters: {'depth': 10, 'learning_rate': 0.1054443423428026, 'n_estimators': 168, 'l2_leaf_reg': 0.008424931568526825, 'border_count': 119, 'scale_pos_weight': 1.0326113617110544}. Best is trial 9 with value: 0.9828631086581128.
[I 2024-11-09 15:36:55,808] Trial 13 finished with value: 0.9764716064291958 and parameters: {'depth': 10, 'learning_rate': 0.07439181019963882, 'n_estimators': 232, 'l2_leaf_reg': 0.009061887352559028, 'border_count': 112, 'scale_pos_weight': 3.3430730176037837}. Best is trial 9 with value: 0.9828631086581128.
[I 2024-11-09 15:36:56,519] Trial 14 finished with value: 0.9815282624049885 and parameters: {'depth': 9, 'learning_rate': 0.05047816600877884, 'n_estimators': 268, 'l2_leaf_reg': 0.19230754683874204, 'border_count': 68, 'scale_pos_weight': 3.4764756572601923}. Best is trial 9 with value: 0.9828631086581128.
[I 2024-11-09 15:36:56,861] Trial 15 finished with value: 0.9655254804555026 and parameters: {'depth': 7, 'learning_rate': 0.028142597770888785, 'n_estimators': 270, 'l2_leaf_reg': 0.257690891803899, 'border_count': 61, 'scale_pos_weight': 3.7089796531443415}. Best is trial 9 with value: 0.9828631086581128.
[I 2024-11-09 15:36:57,611] Trial 16 finished with value: 0.9841153138990156 and parameters: {'depth': 9, 'learning_rate': 0.04771817510352225, 'n_estimators': 267, 'l2_leaf_reg': 0.10954369594151343, 'border_count': 76, 'scale_pos_weight': 2.846132103747829}. Best is trial 16 with value: 0.9841153138990156.
[I 2024-11-09 15:36:57,995] Trial 17 finished with value: 0.9712820475700483 and parameters: {'depth': 7, 'learning_rate': 0.01671697126432103, 'n_estimators': 276, 'l2_leaf_reg': 0.06040114954592021, 'border_count': 80, 'scale_pos_weight': 2.8215282367035948}. Best is trial 16 with value: 0.9841153138990156.
[I 2024-11-09 15:36:58,228] Trial 18 finished with value: 0.9769566510998972 and parameters: {'depth': 6, 'learning_rate': 0.044504470688136934, 'n_estimators': 220, 'l2_leaf_reg': 0.02809995781289414, 'border_count': 32, 'scale_pos_weight': 2.1661441369832066}. Best is trial 16 with value: 0.9841153138990156.
[I 2024-11-09 15:36:59,326] Trial 19 finished with value: 0.9802704381766986 and parameters: {'depth': 9, 'learning_rate': 0.017786385858740802, 'n_estimators': 283, 'l2_leaf_reg': 0.08039252363920976, 'border_count': 136, 'scale_pos_weight': 2.618793587612245}. Best is trial 16 with value: 0.9841153138990156.

catboost best hyperparameters:
{'depth': 9, 'learning_rate': 0.04771817510352225, 'n_estimators': 267, 'l2_leaf_reg': 0.10954369594151343, 'border_count': 76, 'scale_pos_weight': 2.846132103747829}

CatBoost - Best threshold based on f1 score: 0.0172

CatBoost - Classification report:
              precision    recall  f1-score   support

           0     0.9636    0.5668    0.7138       187
           1     0.1000    0.6923    0.1748        13

    accuracy                         0.5750       200
   macro avg     0.5318    0.6296    0.4443       200
weighted avg     0.9075    0.5750    0.6788       200

CatBoost - Confusion matrix:
[[106  81]
 [  4   9]]
CatBoost - ROC AUC score: 0.6236
CatBoost - Average precision score: 0.0994

model comparison:
           Model  Precision    Recall  F1-Score   ROC AUC  Average Precision
0        XGBoost   0.102273  0.692308  0.178218  0.601810           0.092586
1  Random Forest   0.136364  0.461538  0.210526  0.659811           0.103616
2       LightGBM   0.114754  0.538462  0.189189  0.633073           0.096782
3       CatBoost   0.100000  0.692308  0.174757  0.623612           0.099401

best model saved: Random Forest_fraud_detection_model.pkl

new data predictions using Random Forest:
   fraud_probability  is_fraud
0           0.158897         0
1           0.110129         0
2           0.140966         0
3           0.189276         0
4           0.321380         1

$ 
```
