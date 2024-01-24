import numpy as np
import matplotlib.pyplot as plt

# read the header part of the first snapshot
f = open("out0000.txt")
t = float( (f.readline()).split()[-1])
resx = int((f.readline()).split()[-1])
resy = int((f.readline()).split()[-1])
resz = int((f.readline()).split()[-1])
f.close()

# Create a series of png images
for i_s in range(0, 101):#201):


  infile = "out%04d.txt" % i_s

  f = open(infile)
  t = float( (f.readline()).split()[-1])
  f.close()

  data = np.loadtxt(infile, delimiter=', ')

  X = np.array(data[:,0]).reshape(resx, resy, resz)
  Y = np.array(data[:,1]).reshape(resx, resy,resz)
  rho = np.array(data[:,3]).reshape(resx, resy, resz)
  v_y = np.array(data[:,9]).reshape(resx, resy, resz)
  pres = np.array(data[:,10]).reshape(resx, resy, resz)

  fig, ax = plt.subplots()
  ax.axis('equal')
  pcm = ax.pcolormesh(X[:,:,0], Y[:,:,0], rho[:,:,0])#, vmin=0.9, vmax=2.1)
  #pcm = ax.pcolormesh(X, Y, rho)
  fig.colorbar(pcm, ax=ax, extend='max')
  ax.set_title('i = %04d, t = %1.3e' % (i_s, t))

  plt.draw()
  plot_filename = "snapshot%04d.png" % i_s
  plt.savefig(plot_filename)
  plt.close()
  print("saved figure %s" % plot_filename)


#plt.show()




