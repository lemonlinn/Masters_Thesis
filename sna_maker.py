import tweepy
import pandas as pd
import config
import flatten_json
from itertools import combinations
from time import strptime, mktime

import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english')) 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# set up 0Auth
auth = tweepy.OAuthHandler(config.ckey, config.csecret)
auth.set_access_token(config.atoken, config.asecret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

class twitter_SNA(object):
	def __init__(self):
		"""initializes some self.variables"""
		print("Let's get started!")
		self.V1 = list()
		self.V2 = list()
		self.nw = ['coverup', 'pharma', 'sheep', 'corrupt', 'politicians', 
		'corporations', 'politician', 'corporation', 'fever', 'rash', 'rashes', 
		'fevers', 'truth', 'risk', 'risky', 'harm', 'harmful', 'harms', 'injury',
		'system', 'protect', 'profit', 'profits', 'profitable', 'unvaccinated', 
		'vaxxed', 'cdc', 'fda', 'government', 'govt', 'media', 'child', 'children',
		'kid', 'kids', 'mercury', 'thimerosal', 'pertussis', 'effects', 'effect', 'flu',
		'shots', 'shot', 'mainstream', 'homeschool', 'mandatory', 'required',
		'risky', 'risk', 'sick']
		self.pw = ['provax', 'provaxx', 'herd', 'immunity', 'immune', 'doctor', 'doctors', 
		'medical', 'professional', 'science', 'scientific', 'evidence', 'study', 
		'studies', 'law', 'laws', 'health', 'healthy', 'misinformation', 'inform', 
		'informed', 'information', 'source', 'sources', 'autism', 'ableist', 
		'ableism', 'parents', 'parent', 'measles', 'community', 'religious', 
		'religion', 'disease', 'cure', 'cures', 'meningococcal', 'mmr', 'test', 'tested',
		'prove', 'proven', 'shown', 'tests', 'educated', 'misinformed', 'trial', 'trials',
		'adverse', 'mandated']
		#self.pre_df = pd.DataFrame()
		#self.post_df = pd.DataFrame()

	def checkFollow(self, sourceid, targetid):
		"""Checks if two twitter ids are following each other"""
		V1 = self.V1
		V2 = self.V2
		if type(sourceid) is str:
			relation = api.show_friendship(source_screen_name=sourceid, 
				target_screen_name=targetid)
			if relation[0].following and not relation[1].following:
				V1.append(sourceid)
				V2.append(targetid)
			elif relation[1].following and not relation[0].following:
				V1.append(targetid)
				V2.append(sourceid)
			elif relation[0].following and relation[1].following:
				V1.append(targetid)
				V2.append(sourceid)
				V1.append(sourceid)
				V2.append(targetid)
		elif type(sourceid) is int:
			relation = api.show_friendship(source_id=sourceid, 
				target_id=targetid)
			if relation[0].following and not relation[1].following:
				V1.append(sourceid)
				V2.append(targetid)
			elif relation[1].following and not relation[0].following:
				V1.append(targetid)
				V2.append(sourceid)
			elif relation[0].following and relation[1].following:
				V1.append(targetid)
				V2.append(sourceid)
				V1.append(sourceid)
				V2.append(targetid)

	def gen_senti(self, string):
		"""Very simple approx. of sentiment using a handmade corpus"""
		nc = 0
		pc = 0

		clean = re.sub('[^A-Za-z]+', " ", string)
		clean = clean.lower()

		for word in word_tokenize(clean):
			if word in self.nw:
				nc -= 1
			elif word in self.pw:
				pc += 1

		if pc == 0 and nc == 0:
			return(None)
		else:
			S = (nc + pc)/(abs(nc) + pc)

		return(S)

	def edges(self, twit_id, id_list):
		"""Collects edgelist of follower/following directional ties"""
		if len(id_list) != 0:
			print("Number of retweeters given: " + str(len(id_list)))
			#retweeter_names = []
			#for i in id_list:
			#	retweeter_names.append(re.sub("(@)", "", i))
			retweeter_names = id_list

		else:
			target_retweeters = api.retweeters(twit_id)
			print("Number of retweeters collected: " + str(len(target_retweeters)))
			retweeter_names = [api.get_user(r)._json['screen_name'] for r in target_retweeters]

		combos = list(combinations(retweeter_names, 2))
		cnt = 0
		for l in combos:
			cnt += 1
			print("checking combination #" + str(cnt) + " out of " + str(len(combos)))
			self.checkFollow(l[0], l[1])

		edges = pd.DataFrame({"V1": self.V1, "V2": self.V2})
		return(edges)

	def get_pos(self, word):
		"""Tags words with parts-of-speech (POS) label for lemmatization"""
		tag = nltk.pos_tag([word])[0][1][0].upper()
		tag_dict = {"J": wordnet.ADJ,
					"N": wordnet.NOUN,
					"V": wordnet.VERB,
					"R": wordnet.ADV}

		return tag_dict.get(tag, wordnet.NOUN)

	def kleenex(self, var):
		"""Cleans text for processing"""
		tmp = re.sub('(https://t.co/).*', ' ', var)
		tmp = re.sub('@.+?\s', ' ', tmp)
		tmp = re.sub('[^a-zA-Z]+', ' ', tmp)
		tmp = re.sub('^(RT)', ' ', tmp)
		tmp = tmp.lower()
		return(tmp) #used to be clean_tweet

	def countNLP(self, twit_id, nodes):
		"""Uses logistic regression classification modeling to predict sentiment"""
		print("starting to count anti- and pro-vaxx words...")
		pre_col = []
		post_col = []
		pre_NLP = []
		post_NLP = []
		d = pd.read_csv("thesis_NLP_ML_data_2020-06-19.csv") # test/train data collected through Twitter_NLP_Model.ipynb
		X = d.loc[:,d.columns != "label"]
		y = d["label"]

		tfidf_vec = TfidfVectorizer()
		lemmer = WordNetLemmatizer()
		X_train = tfidf_vec.fit_transform(X["lemma_sw"].values.astype('U'))

		logreg = LogisticRegression(C=1, solver='liblinear', penalty = 'l2', max_iter=100000, random_state=42)
		logreg.fit(X_train, y)

		for name in nodes.names:
			try:
				print("Checking sentiment of " + name)
				post_event = list(tweepy.Cursor(api.user_timeline, id='{}'.format(name), since_id=twit_id, tweet_mode='extended').items())
				pre_event = list(tweepy.Cursor(api.user_timeline, id='{}'.format(name), max_id=twit_id, tweet_mode='extended').items(len(post_event)))
				pre_flat = []
				post_flat = []

				for i in range(0,len(post_event)):
					post_flat.append(flatten_json.flatten_json(post_event[i]._json))
				for i in range(0, len(pre_event)):
					pre_flat.append(flatten_json.flatten_json(pre_event[i]._json))

				pre_df = pd.json_normalize(pre_flat)
				post_df = pd.json_normalize(post_flat)
				presenti = []
				postsenti = []
				pre_lemmed = []
				post_lemmed = []
				for tweet in pre_df.full_text:
					presenti.append(self.gen_senti(tweet))
					if self.gen_senti(tweet) is not None:
						tmp = self.kleenex(tweet)
						tmp = word_tokenize(tmp)
						tmp = [lemmer.lemmatize(wrd, self.get_pos(wrd)) for wrd in tmp if wrd not in stop_words]
						pre_lemmed.append(" ".join(tmp))
					else:
						pre_lemmed.append(None)

				try:
					presenti = list(filter(None, presenti))
					pre_sum = sum(presenti)/len(presenti)

					pre_train = tfidf_vec.transform(list(filter(None, pre_lemmed)))
					pre_preds = logreg.predict(pre_train)
					pre_final = sum(pre_preds)/len(pre_preds)
				except ZeroDivisionError:
					pre_sum = None
					pre_final = None

				for tweet in post_df.full_text:
					postsenti.append(self.gen_senti(tweet))
					if self.gen_senti(tweet) is not None:
						tmp = self.kleenex(tweet)
						tmp = word_tokenize(tmp)
						tmp = [lemmer.lemmatize(wrd, self.get_pos(wrd)) for wrd in tmp if wrd not in stop_words]
						post_lemmed.append(" ".join(tmp))
					else:
						post_lemmed.append(None)

				try:
					postsenti = list(filter(None, postsenti))
					post_sum = sum(postsenti)/len(postsenti)

					post_train = tfidf_vec.transform(list(filter(None, post_lemmed)))
					post_preds = logreg.predict(post_train)
					post_final = sum(post_preds)/len(post_preds)
				except ZeroDivisionError:
					post_sum = None
					post_final = None

				pre_col.append(pre_sum)
				post_col.append(post_sum)
				pre_NLP.append(pre_final)
				post_NLP.append(post_final)

			except AttributeError:
				print("Something went wrong with " + name)
				pre_col.append(None)
				post_col.append(None)
				pre_NLP.append(None)
				post_NLP.append(None)
				continue

		diff_col = []
		diff_NLP = []
		for pre,post in zip(pre_col, post_col):
			if pre is not None and post is not None:
				if pre >= post:
					diff_col.append(abs(pre - post)*-1)
				else:
					diff_col.append(abs(post - pre))
			else:
				diff_col.append(None)

		for pre,post in zip(pre_NLP, post_NLP):
			if pre is not None and post is not None:
				diff_NLP.append(post-pre)
			else:
				diff_NLP.append(None)

		return(pre_col, post_col, diff_col, pre_NLP, post_NLP, diff_NLP)

	def main(self, twit_id, id_list = []):
		"""Main function call, as well as the node attributes collection"""
		print("starting to collect edges...")
		edges = self.edges(twit_id, id_list = id_list)
		V3 = set(edges.V1)

		print("starting to collect node attributes...")
		for v in edges.V2:
			V3.add(v)

		V3 = list(V3)
		nodes = pd.DataFrame({"names":list(set(V3))})
		
		V4 = [api.get_user(i)._json for i in V3]
		favourites_count = [j["favourites_count"] for j in V4]
		lang = [j["lang"] for j in V4]
		followers_count = [j["followers_count"] for j in V4]
		description = [j["description"] for j in V4]
		friends_count = [j["friends_count"] for j in V4]
		listed_count = [j["listed_count"] for j in V4]
		statuses_count = [j["statuses_count"] for j in V4]
		uuid = [j["id"] for j in V4]

		nodes['uuid'] = uuid
		nodes['favourites_count'] = favourites_count
		nodes['lang'] = lang
		nodes['followers_count'] = followers_count
		nodes['description'] = description
		nodes['friends_count'] = friends_count
		nodes['listed_count'] = listed_count
		nodes['statuses_count'] = statuses_count

		pre_col, post_col, diff_col, pre_NLP, post_NLP, diff_NLP = self.countNLP(twit_id, nodes)
		nodes['pre_sent'] = pre_col
		nodes['post_sent'] = post_col
		nodes['diff_sent'] = diff_col
		nodes['pre_ML'] = pre_NLP
		nodes['post_ML'] = post_NLP
		nodes['diff_ML'] = diff_NLP

		return(nodes, edges)

		