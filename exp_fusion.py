import tensorflow as tf

def saturation(img):
	mean = tf.keras.backend.mean(img, axis=-1, keepdims = True)
	mul = tf.constant([1,1,1,3], tf.int32)
	mean = tf.tile(mean, mul)
	img = tf.math.subtract(img, mean)
	sat = tf.einsum('aijk,aijk->aij', img, img)
	sat = tf.math.scalar_mul((1.0/3.0),sat)
	sat = tf.math.sqrt(sat)
	return sat

def get_exp(img,c):
	cimg = tf.slice(img,[0,0,0,c],[img.get_shape()[0],img.get_shape()[1],img.get_shape()[2],1])
	cimg = tf.squeeze(cimg,axis=-1)
	m = tf.math.scalar_mul(0.5, tf.ones_like(cimg))
	cimg = tf.math.subtract(cimg,m)
	cimg = tf.math.multiply(cimg,cimg)
	cimg = tf.math.scalar_mul(-12.5,cimg)
	return cimg

def exposure(img):
	rimg = get_exp(img,0)
	gimg = get_exp(img,1)
	bimg = get_exp(img,2)
	img = tf.math.add(rimg,gimg)
	img = tf.math.add(img,bimg)
	exp = tf.math.exp(img)
	return exp

def contrast(img):
	mean = tf.keras.backend.mean(img, axis=-1, keepdims=True)
	lap_fil = [[0,-1,0],[-1,4,-1],[0,-1,0]]
	lap_fil = tf.expand_dims(lap_fil,-1)
	lap_fil = tf.expand_dims(lap_fil,-1)
	con = tf.nn.convolution(mean, lap_fil, padding='SAME')
	con = tf.math.abs(con)
	con = tf.squeeze(con,axis=-1)
	return con

def exp_map(img,pc,ps,pe):
	con = contrast(img)
	sat = saturation(img)
	exp = exposure(img)
	if pc!=1 or pe!=1 or ps!=1:
		pc = tf.math.scalar_mul(pc, tf.ones_like(con))
		ps = tf.math.scalar_mul(ps, tf.ones_like(con))
		pe = tf.math.scalar_mul(pe, tf.ones_like(con))
		con = tf.math.pow(con,pc)
		sat = tf.math.pow(sat,pe)
		exp = tf.math.pow(exp,ps)
	wt_map = tf.math.multiply(con,sat)
	wt_map = tf.math.multiply(wt_map,exp)
	return wt_map
