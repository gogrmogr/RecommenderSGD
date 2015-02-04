import random
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from numpy import dtype

#gamma - learning rate
#lambda - regularization const
#numbers correspond to equations in netflix paper
class SGDRecommender():
	def __init__(self, read_path, load_existing = False, n_features = 5, lambda1 = 0.007, lambda2 = 0.007, gamma1 = 0.02, gamma_factor = 0.9, max_iter1 = 1, max_iter2 = 1, init_mean = 0.1, init_stdev = 0.1):
		if load_existing == True:
			self.ReadModel(read_path)
			return
		
		self.rating_mtrx_file = read_path
		self.n_features = n_features
		self.lambda1 = lambda1
		self.lambda2 = lambda2
		self.gamma1 = gamma1
		self.gamma_factor = gamma_factor
		self.max_iter1 = max_iter1
		self.max_iter2 = max_iter2
		self.init_mean = init_mean
		self.init_stdev = init_stdev
		
		self.mu = None
		
		self.train_rmse = None
		
		self.bu = None
		self.bi = None
		self.bui = None
		self.V_train = None
		self.V_train_nidx = None
		self.V_test = None
		self.V_test_nidx = None
		self.user_list = None
		self.bus_list = None
		self.user_dict = None
		self.bus_dict = None

		self.q_factors = None
		self.x_factors = None
		self.y_factors = None
		
		self.LoadMatrix()
		self.MakeTrainTest()
	
	def ReadModel(self, dir):
		print ("Reading a model from disk...")
		if os.path.isdir(dir):
			os.chdir(dir)
		else:
			print ("Path {} is not a directory. Model was not loaded.".format(dir))
			return
		
		all_models = [file.split('.')[0] for file in os.listdir(dir) if file.endswith(".settings")]
		if not len(all_models):
			print ("No models were found in directory {}. Model was not loaded".format(dir))
			return

		file_prefix = max(all_models)
    			
		f_i = open("{}.settings".format(file_prefix))
		self.rating_mtrx_file = f_i.readline().rstrip().split()[1]
		self.n_features = int(f_i.readline().rstrip().split()[1])
		self.lambda1 = float(f_i.readline().rstrip().split()[1])
		self.lambda2 = float(f_i.readline().rstrip().split()[1])
		self.gamma1 = float(f_i.readline().rstrip().split()[1])
		self.gamma_factor = float(f_i.readline().rstrip().split()[1])
		self.max_iter1 = int(f_i.readline().rstrip().split()[1])
		self.max_iter2 = int(f_i.readline().rstrip().split()[1])
		self.init_mean = float(f_i.readline().rstrip().split()[1])
		self.init_stdev = float(f_i.readline().rstrip().split()[1])
		f_i.close()

		f_i = open("{}.rmse".format(file_prefix))
		self.train_rmse = float(f_i.readline().rstrip())
		f_i.close()

		bu_f = "{}_bu.bin".format(file_prefix)
		bi_f = "{}_bi.bin".format(file_prefix)
		bui_f = "{}_bui.bin".format(file_prefix)
		V_train_f = "{}_V_train.bin".format(file_prefix)
		V_test_f = "{}_V_test.bin".format(file_prefix)
		q_f = "{}_q.bin".format(file_prefix)
		x_f = "{}_x.bin".format(file_prefix)
		y_f = "{}_y.bin".format(file_prefix)
		user_dict_f = "{}_users.json".format(file_prefix)
		bus_dict_f = "{}_businesses.json".format(file_prefix)

		f_i = open(bu_f, "rb")
		self.bu = np.load(f_i)
		f_i.close()
		f_i = open(bi_f, "rb")
		self.bi = np.load(f_i)
		f_i.close()
		f_i = open(bui_f, "rb")
		self.bui = np.load(f_i)
		f_i.close()
		f_i = open(V_train_f, "rb")
		self.V_train = np.load(f_i)
		f_i.close()
		f_i = open(V_test_f, "rb")
		self.V_test = np.load(f_i)
		f_i.close()
		f_i = open(q_f, "rb")
		self.q_factors = np.load(f_i)
		f_i.close()
		f_i = open(x_f, "rb")
		self.x_factors = np.load(f_i)
		f_i.close()
		f_i = open(y_f, "rb")
		self.y_factors = np.load(f_i)
		f_i.close()
		f_i = open(user_dict_f)
		self.user_dict = json.load(f_i)
		f_i.close()
		f_i = open(bus_dict_f)
		self.bus_dict = json.load(f_i)
		f_i.close()
		
		self.user_list = [x[0] for x in sorted(self.user_dict.items(), key=lambda x: x[1])]
		self.bus_list = [x[0] for x in sorted(self.bus_dict.items(), key=lambda x: x[1])]

		mu = 0.0
		self.V_train_nidx = ([], [])
		for i in range(len(self.V_train)):
			for j in self.V_train[i]:
				self.V_train_nidx[0].append(i)
				self.V_train_nidx[1].append(j)
				mu += self.V_train[i][j]
		self.mu = float(mu)/sum([len(x) for x in self.V_train])
		
		self.V_test_nidx = ([], [])
		for i in range(len(self.V_test)):
			for j in self.V_test[i]:
				self.V_test_nidx[0].append(i)
				self.V_test_nidx[1].append(j)

	def WriteModel(self):
		print ("Writing the model on disk...")
		os.mkdir("./model")
		os.chdir("./model")
		file_prefix = datetime.now().strftime('%Y%m%d%H%M%S')
		
		f_o = open("{}.settings".format(file_prefix), "w")
		f_o.write("rating_mtrx_file {}\n".format(self.rating_mtrx_file))
		f_o.write("n_features {}\n".format(self.n_features))
		f_o.write("lambda1 {}\n".format(self.lambda1))
		f_o.write("lambda2 {}\n".format(self.lambda2))
		f_o.write("gamma1 {}\n".format(self.gamma1))
		f_o.write("gamma_factor {}\n".format(self.gamma_factor))
		f_o.write("max_iter1 {}\n".format(self.max_iter1))
		f_o.write("max_iter2 {}\n".format(self.max_iter2))
		f_o.write("init_mean {}\n".format(self.init_mean))
		f_o.write("init_stdev {}\n".format(self.init_stdev))
		f_o.close()

		f_o = open("{}.rmse".format(file_prefix), "w")
		f_o.write("{}\n".format(self.train_rmse))
		f_o.close()

		bu_f = "{}_bu.bin".format(file_prefix)
		bi_f = "{}_bi.bin".format(file_prefix)
		bui_f = "{}_bui.bin".format(file_prefix)
		V_train_f = "{}_V_train.bin".format(file_prefix)
		V_test_f = "{}_V_test.bin".format(file_prefix)
		q_f = "{}_q.bin".format(file_prefix)
		x_f = "{}_x.bin".format(file_prefix)
		y_f = "{}_y.bin".format(file_prefix)
		user_dict_f = "{}_users.json".format(file_prefix)
		bus_dict_f = "{}_businesses.json".format(file_prefix)

		f_o = open(bu_f, "wb")
		np.save(f_o, self.bu)
		f_o.close()
		f_o = open(bi_f, "wb")
		np.save(f_o, self.bi)
		f_o.close()
		f_o = open(bui_f, "wb")
		np.save(f_o, self.bui)
		f_o.close()
		f_o = open(V_train_f, "wb")
		np.save(f_o, self.V_train)
		f_o.close()
		f_o = open(V_test_f, "wb")
		np.save(f_o, self.V_test)
		f_o.close()
		f_o = open(q_f, "wb")
		np.save(f_o, self.q_factors)
		f_o.close()
		f_o = open(x_f, "wb")
		np.save(f_o, self.x_factors)
		f_o.close()
		f_o = open(y_f, "wb")
		np.save(f_o, self.y_factors)
		f_o.close()
		f_o = open(user_dict_f, "w")
		json.dump(self.user_dict, f_o)
		f_o.close()
		f_o = open(bus_dict_f, "w")
		json.dump(self.bus_dict, f_o)
		f_o.close()

	def GetServerUserResult(self, u, how_many = -1):
		rmses = self.GetUserTopN(u, how_many)
		user_mtrx = []

		for elem in rmses:
			row_list = []
			row_list.extend([elem[0], elem[1]])
			importances = sorted(self.GetAllImportant(u, elem[0]).items(), key=lambda x: x[1], reverse = True)
			for i in importances:
				row_list.append(i[0])
			user_mtrx.append(row_list)
		return user_mtrx
	
	def WriteUserResult(self, f_o, u):
		rmses = self.GetUserTopN(u, -1)
		for elem in rmses:
			importances = sorted(self.GetAllImportant(u, elem[0]).items(), key=lambda x: x[1], reverse = True)
			all_j = [i[0] for i in importances]
			s = "{} {:.4}".format(elem[0], elem[1])
			for i in all_j:
				s = s + " {}".format(i)
			s = s + "\n"
			f_o.write(s)

	def WriteServerResults(self, u_range = (0,0)):
		if not os.path.isdir("./UserResults"):
			os.mkdir("./UserResults")
		os.chdir("./UserResults")
		f_o = open("BusinessList.txt", "w")
		for i in self.bus_list:
			f_o.write("{}\n".format(i))
		f_o.close()
		if (u_range[1] - u_range[0]) < 1:
			start = 0
			end = len(self.user_list)
		else:
			start = u_range[0]
			end = u_range[1]
			
		for u in range(start, end):
			if os.path.isfile(self.user_list[u]):
				continue
			f_o = open(self.user_list[u], "w")
			self.WriteUserResult(f_o, u)
			f_o.close()		
		os.chdir("../")	
		
	def LoadMatrix1(self):
		print ("Loading data from rating file...")
		if not os.path.isfile(self.rating_mtrx_file):
			print ("{} is not a file. Model was not built".format(self.rating_mtrx_file))
			return
		mtrx_file = open(self.rating_mtrx_file)
		tmp_lst = []
		for ln in mtrx_file:
			tmp_lst.append(json.loads(ln))
		self.ratings_df = pd.DataFrame(tmp_lst)
		mtrx_file.close()
		
	def LoadMatrix(self):
		print ("Loading data from rating file...")
		if not os.path.isfile(self.rating_mtrx_file):
			print ("{} is not a file. Model was not built".format(self.rating_mtrx_file))
			return
		mtrx_file = open(self.rating_mtrx_file)
		self.ratings_df = pd.io.json.read_json(mtrx_file, orient = 'records')
		mtrx_file.close()
	
	#to read from 3 column text files
	def LoadMatrix2(self):
		print ("Loading data from rating file...")
		if not os.path.isfile(path):
			print ("{} is not a file. Model was not built".format(path))
			return
		mtrx_file = open(self.rating_mtrx_file)
		tmp_lst = []
		for ln in mtrx_file:
			tmp_lst.append(ln.rstrip().split())
		self.ratings_df = pd.DataFrame(tmp_lst)
		self.ratings_df.columns = ["user_id", "business_id", "stars"]
		mtrx_file.close()
		
	#pick a random 1/n-th of filtered reviews for the training set. If nusers parameter is given, choose only reviews from random <nusers> users before splitting. In test set include the remaining reviews but remove users and businesses that didn't appear in training set. Test set usually has rating due to odd number of ratings for some users
	def _random_split_test_train(reviews, n = 1.4, nusers = 0):
	    #get indices for random half of the review set
	    if not nusers:
	        filtered_reviews = reviews
	    else:
	        users_test = random.sample(list(reviews["user_id"].unique()), nusers)
	        filtered_reviews = reviews[reviews["user_id"].isin(users_test)]
	
	    grouped = filtered_reviews.groupby("user_id")
	    reviews_train = []
	    for g in grouped.groups:
	        reviews_train.extend(random.sample(grouped.groups[g], int(len(grouped.groups[g])/n)))
	
	    train_df = filtered_reviews[filtered_reviews.index.isin(reviews_train)]
	    test_df = filtered_reviews[~filtered_reviews.index.isin(reviews_train)]
	
	    return train_df, test_df

	#given column of df (either "user_id" or "business_id") make a dictionary of id to number. Make sure df contains all users and businesses (e.g. train_df.append(test_df))
	def _make_dict(df, column):
	    new_dict = {}
	    names = df[column].unique()
	    for i in range(len(names)):
	        new_dict[names[i]] = i
	
	    return new_dict
	   
	def _get_rating_idxs(self, df):
		#V = sp.lil_matrix((len(user_dict),len(bus_dict)))
		V = [{} for _ in range(len(self.user_dict))]
		nidx = ([], [])
		for index, row in df.iterrows():
			i = self.user_dict[row['user_id']]
			j = self.bus_dict[row['business_id']]
			nidx[0].append(i)
			nidx[1].append(j)
			V[i][j] = row['stars']
	
		return V, nidx

	def MakeTrainTest(self):
		print ("Splitting data into training and test sets...")
		#make a dictionary to map from index to user_id and business_id
		self.user_dict = self._make_dict(self.ratings_df, "user_id")
		self.bus_dict = self._make_dict(self.ratings_df, "business_id")
		self.user_list = [x[0] for x in sorted(self.user_dict.items(), key=lambda x: x[1])]
		self.bus_list = [x[0] for x in sorted(self.bus_dict.items(), key=lambda x: x[1])]
		
		train_df, test_df = self._random_split_test_train(self.ratings_df) 
		self.V_train, self.V_train_nidx = self._get_rating_idxs(train_df)
		self.V_test, self.V_test_nidx = self._get_rating_idxs(test_df)

		#self.V_train = normalize_mat_by_max(V_train)
		#self.V_test = normalize_mat_by_max(V_test)
		self.bui = [{} for _ in range(len(self.user_dict))]
		mu = 0
		for i in range(len(self.V_train)):
			for j in self.V_train[i]:
				self.bui[i][j] = 0
				mu += self.V_train[i][j]
		self.mu = float(mu)/sum([len(x) for x in self.V_train])

	def _predict(self, u, i):
		p = self.mu + self.bi[i] + self.bu[u]
		
		sum_xj = sum_yj = np.zeros(self.n_features, dtype = float)
		if self.y_factors == None:
			return p, sum_xj, sum_yj
		if not len(self.V_train[u].keys()):
			return p, sum_xj, sum_yj

		coef = 1.0/np.sqrt(len(self.V_train[u].keys()))
		
		for j in self.V_train[u].keys():
			sum_yj += self.y_factors[j]
			sum_xj += (self.V_train[u][j] - self.bui[u][j]) * self.x_factors[j]
		
		p += np.dot(self.q_factors[i], coef * (sum_xj + sum_yj))
			
		return p, sum_xj, sum_yj
	
	def _trainBui_iter(self, shuffled_indices, current_gamma):
		nidx = self.V_train_nidx
		total_err = 0.0
		
		for elem in shuffled_indices:
			u = nidx[0][elem]
			i = nidx[1][elem]

			err = self.V_train[u][i] - self.mu - self.bu[u] - self.bi[i]
			total_err += (err ** 2)
			self.bu[u] = self.bu[u] + current_gamma * (err - self.lambda1 * self.bu[u])
			self.bi[i] = self.bi[i] + current_gamma * (err - self.lambda1 * self.bi[i])
			self.bui[u][i] = self.mu + self.bu[u] + self.bi[i]

		return total_err
			
	def TrainBui1(self, init = True):
		print ("Training bui's...")
		nidx = self.V_train_nidx
		shuffled_indices = list(range(0, len(nidx[0])))
		random.shuffle(shuffled_indices)
		if init:
			self.bu = self.init_mean * np.random.randn(len(self.user_dict)) + self.init_stdev ** 2
			self.bi = self.init_mean * np.random.randn(len(self.bus_dict)) + self.init_stdev ** 2
			current_gamma = self.gamma1
		else:
			current_gamma = self.gamma1 * (self.gamma_factor ** self.max_iter1)
		for i in range(self.max_iter1):
			err = self._trainBui_iter(shuffled_indices, current_gamma)
			rmse = np.sqrt(err / len(shuffled_indices))
			print("Finished the interation {} with RMSE {}".format(i, rmse))
			current_gamma *= self.gamma_factor
		
	def _trainFactors_iter(self, shuffled_indices, current_gamma):
		nidx = self.V_train_nidx
		total_err = 0.0
		
		for elem in shuffled_indices:
			u = nidx[0][elem]
			i = nidx[1][elem]
			p, sum_xj, sum_yj = self._predict(u, i)
			err = self.V_train[u][i] - p
			total_err += (err ** 2)
			
			coef = 1.0/np.sqrt(len(self.V_train[u].keys()))
			self.bu[u] = self.bu[u] + current_gamma * (err - self.lambda1 * self.bu[u])
			self.bi[i] = self.bi[i] + current_gamma * (err - self.lambda1 * self.bi[i])
			
			for j in self.V_train[u].keys():
				delta_x = current_gamma * (err * coef * self.q_factors[i] * (self.V_train[u][j] - self.bui[u][j]) - self.lambda2 * self.x_factors[j])
				delta_y = current_gamma * (err * coef * self.q_factors[i] - self.lambda2 * self.y_factors[j])
				self.x_factors[j] += delta_x
				self.y_factors[j] += delta_y

			delta_q = current_gamma * (err * coef * (sum_xj + sum_yj) - self.lambda2 * self.q_factors[i])
			self.q_factors[i] += delta_q
			
		return total_err

	def TrainFactors13(self, init = True):
		print ("Training factors...")
		nidx = self.V_train_nidx
		shuffled_indices = list(range(0, len(nidx[0])))
		random.shuffle(shuffled_indices)
		#self.bu = self.init_mean * np.random.randn(len(user_dict)) + self.init_stdev ** 2
		#self.bi = self.init_mean * np.random.randn(len(bus_dict)) + self.init_stdev ** 2
		if init:
			self.q_factors = self.init_mean * np.random.randn(len(self.bus_dict), self.n_features) + self.init_stdev ** 2
			self.x_factors = self.init_mean * np.random.randn(len(self.bus_dict), self.n_features) + self.init_stdev ** 2
			self.y_factors = self.init_mean * np.random.randn(len(self.bus_dict), self.n_features) + self.init_stdev ** 2
			current_gamma = self.gamma1
		else:
			current_gamma = self.gamma1 * (self.gamma_factor ** self.max_iter2)
		for i in range(self.max_iter2):
			err = self._trainFactors_iter(shuffled_indices, current_gamma)
			self.train_rmse = np.sqrt(err / len(shuffled_indices))
			print("Finished interation {} with RMSE {}".format(i, self.train_rmse))
			current_gamma *= self.gamma_factor

	def GetAllImportant(self, u, i):
		j_importance = {}
		for j in self.V_train[u].keys():
			j_importance[j] = np.dot(self.q_factors[i], (self.V_train[u][j] - self.bui[u][j]) * self.x_factors[j] + self.y_factors[j])
		
		return j_importance
	
	def GetMostImportant(self, u, i):
		all_j = self.GetAllImportant(u, i)
		if all_j:
			return sorted(all_j.items(), key=lambda x: x[1], reverse = True)[0][0]
		return -1
		
		
	def GetUserTopN(self, u, n):
		all_ratings = []
		for i in range(len(self.bus_dict)):
			p, _1, _2 = self._predict(u, i)
			all_ratings.append(p)
			
		sorted_ratings = [i for i in sorted(enumerate(all_ratings), key=lambda x:x[1], reverse = True)]
		return sorted_ratings[:n]
	
	def RMSETestSet(self):
		print ("Calculating RMSE of the test set...")
		total_err = 0.0		
		for u in range(len(self.V_test)):
			for i in self.V_test[u]:
				p, _1, _2 = self._predict(u, i)
			
				if p > 5.0: p = 5.0
				if p < 1.0: p = 1.0

				err = self.V_test[u][i] - p
				total_err += (err ** 2)
		rmse = np.sqrt(total_err / sum([len(x) for x in self.V_test]))
		
		return rmse
	
	def RMSEPerUser(self, u):
		print ("Calculating RMSE of the user...")
		user_rmses = 0.0
		for i in self.V_test[u]:
			p, _1, _2 = self._predict(u, i)

			if p > 5.0: p = 5.0
			if p < 1.0: p = 1.0

			err = self.V_test[u][i] - p
			user_rmses += (err ** 2)
		user_rmses = np.sqrt(user_rmses/len(self.V_test[u]))

		return user_rmses

	def AddUser(self, u_id, u_ratings):
		self.user_list.append(u_id)
		self.user_dict[u_id] =len(self.user_list) - 1
		
		self.V_train.append({})
		for i, r in u_ratings:
			self.V_train[-1][i] = r
			self.V_train_nidx[0].append(u)
			self.V_train_nidx[1].append(i)
				
		mu = 0.0
		for i in range(len(self.V_train)):
			for j in self.V_train[i]:
				mu += self.V_train[i][j]
		self.mu = float(mu)/sum([len(x) for x in self.V_train])
		
		user_mean = np.mean([r for i, r in u_ratings])
		self.bu = np.append(self.bu, user_mean - self.mu)
		self.TrainBui1(init = False)

