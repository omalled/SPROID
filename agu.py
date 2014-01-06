#This is the code that was used to run the agu examples.

#split to yaml converts the "tracks.txt" file output by I. Jankovic's split
#code into sproid yaml files
def splitToYaml(filename):
	infile = open(filename, "r")
	data = infile.read()
	infile.close()
	bigArray = np.genfromtxt(StringIO(data), delimiter=",")
	numTrajectories = int(bigArray[-1][-1])
	listTrajectories = map(lambda x: [], range(0, numTrajectories))
	for line in bigArray[1:]:
		x = np.asscalar(line[0])
		y = np.asscalar(line[1])
		t = np.asscalar(line[-2])
		trajNum = int(line[-1])
		listTrajectories[trajNum-1].append([t, x, y])
	#listTrajectories = [[[0, 0], [1, 1]], [[0, 0], [1, 1]]]
	data = {"Trajectories": listTrajectories}
	outfile = open(filename + ".yaml", "w")
	yaml.dump(data, outfile)
	outfile.close()

#split to yaml converts the "tracks.txt" file output by I. Jankovic's split
#code into sproid yaml files
def fractureToYaml(filename):
	infile = open(filename, "r")
	data = infile.read()
	infile.close()
	bigArray = np.genfromtxt(StringIO(data), delimiter=" ")
	numTrajectories = 1
	listTrajectories = map(lambda x: [], range(0, numTrajectories))
	for line in bigArray:
		x = np.asscalar(line[2])
		y = np.asscalar(line[3])
		z = np.asscalar(line[4])
		t = np.asscalar(line[-1])
		listTrajectories[0].append([t, x, y, z])
	#listTrajectories = [[[0, 0], [1, 1]], [[0, 0], [1, 1]]]
	data = {"Trajectories": listTrajectories}
	outfile = open(filename + ".yaml", "w")
	yaml.dump(data, outfile)
	outfile.close()

def convertNataliiasData():
	for i in range(1, 2784):
		fractureToYaml("fracture/traject_" + str(i))

def testFracture():
	s = Sproid(filename="traject_1.yaml", downsamples=100)
	traj = s.getTrajectory(0)
	results = traj.aic1D(0, [bm, bmd, fbm, slm])
	return results

def runFracture(filenum, posIndex=0):
	s = Sproid(filename="fracture/traject_" + str(filenum) + ".yaml", downsamples=50)
	traj = s.getTrajectory(0)
	traj = traj.getTrajectory1D(posIndex)
	results = traj.aic1D(0, [bm, bmd, fbm, slm])
	minaic = results[0][0]
	for i in range(1, 4):
		if results[i][0] < minaic:
			minaic = results[i][0]
	outfile = open("fracture/traject_" + str(filenum) + ".results" + str(posIndex), "w")
	for i in range(0, 4):
		if results[i][0] == minaic:
			if i == 0:
				outfile.write("BM\n")
			if i == 1:
				outfile.write("BMD\n")
			if i == 2:
				outfile.write("FBM\n")
			if i == 3:
				outfile.write("SLM\n")
	outfile.write("BM " + str(results[0][0]) + "\n")
	outfile.write(str(results[0][2][0]) + "\n")
	outfile.write("BMD " + str(results[1][0]) + "\n")
	outfile.write(str(results[1][2][0]) + " " + str(results[1][2][1]) + "\n")
	outfile.write("FBM " + str(results[2][0]) + "\n")
	outfile.write(str(results[2][2][0]) + " " + str(results[2][2][1]) + "\n")
	outfile.write("SLM " + str(results[3][0]) + "\n")
	outfile.write(str(results[3][2][0]) + " " + str(results[3][2][1]) + "\n")
	outfile.close()

def runFracture1(i):
	runFracture(i, 1)

def runFracture2(i):
	runFracture(i, 2)

def runFractures():
	n = 2784
	#n = 6
	pool = multiprocessing.Pool(4)
	pool.map(runFracture, range(1, n))
	pool.map(runFracture1, range(1, n))
	pool.map(runFracture2, range(1, n))
	#pool.map(runFracture, range(1, n), map(lambda x: 0, range(1, n)))
	#pool.map(runFracture, range(1, n), map(lambda x: 1, range(1, n)))
	#pool.map(runFracture, range(1, n), map(lambda x: 2, range(1, n)))

#runFractures()

def runSplit():
	n = 100
	pool = multiprocessing.Pool(4)
	pool.map(runSplitSupport, range(0, n))

def runSplitSupport(k):
	s = Sproid(filename="split/tracks.txt.yaml", downsamples=50)
	traj2D = s.getTrajectory(k)
	if traj2D.getPositions1D(0)[-1] > 0:
		for j in range(0, 2):
			traj = traj2D.getTrajectory1D(j)
			results = traj.aic1D(0, [slowbm, slowbmd, slowfbm, slowslm])
			minaic = results[0][0]
			for i in range(1, 4):
				if results[i][0] < minaic:
					minaic = results[i][0]
			outfile = open("split/traject_" + str(k) + ".results" + str(j), "w")
			for i in range(0, 4):
				if results[i][0] == minaic:
					if i == 0:
						outfile.write("BM\n")
					if i == 1:
						outfile.write("BMD\n")
					if i == 2:
						outfile.write("FBM\n")
					if i == 3:
						outfile.write("SLM\n")
			outfile.write("BM " + str(results[0][0]) + "\n")
			outfile.write(str(results[0][2][0]) + "\n")
			outfile.write("BMD " + str(results[1][0]) + "\n")
			outfile.write(str(results[1][2][0]) + " " + str(results[1][2][1]) + "\n")
			outfile.write("FBM " + str(results[2][0]) + "\n")
			outfile.write(str(results[2][2][0]) + " " + str(results[2][2][1]) + "\n")
			outfile.write("SLM " + str(results[3][0]) + "\n")
			outfile.write(str(results[3][2][0]) + " " + str(results[3][2][1]) + "\n")
			outfile.close()


