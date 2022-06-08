import numpy as np

class linear_reg:
	def __init__(self, lr=0.001, n=100):
		self.lr = lr
		self.n = n
		self.w = None
		self.b = None

	def fit(self, x, y):
		n_samp, m_feat = x.shape
		self.w = np.zeros(m_feat)
		self.b = 0

		for _ in range(self.n):
			y_pred = np.dot(x, self.w) + self.b
			dw = (1 / n_samp) * np.dot(x.T, (y_pred - y))
			db = (1 / n_samp) * np.sum(y_pred - y)
			self.w -= self.lr *dw
			self.b -= self.lr *db
		return y_pred

	def predict(self, x):
		y_approx = np.dot(x, self.w) + self.b
		return y_approx
