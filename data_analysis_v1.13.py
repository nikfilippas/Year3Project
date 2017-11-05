import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import ks_2samp
from scipy.stats.distributions import chi2 as chisq
from scipy.optimize import curve_fit
from scipy.constants import c, h
from scipy.constants import k as kB
from astropy.constants import M_sun



def PlanckMod(f, D, M, T, beta):
	"""
	Calculates the grey-body curve of an object at distance D, dust mass M and
	temperature T.
	"""
	from scipy.constants import h, c, parsec
	from scipy.constants import k as kB
	k0 = 0.192; f0 = 856.5e+9  # calibration of modified Planck function

	D = D * 1e+6 * parsec  # convert [Mpc] to [m]
	planck = (2*h*f**3 / c**2) / (np.exp(h*f / (kB*T)) - 1)
	planck_mod = (M/D**2) * k0 * (f/f0)**beta * planck

	return planck_mod



def FiducialCut(dataset, data, col, normcol=("which",None), nbins=3):
	"""
	Returns array of objects split by set property.
	dataset: dataset used | col: column used | nbins: number of bins
	normcol: norm column; indicate `dataset` or `data` and column number
	"""
	DATA = dataset[:,col]
	# property bins (unweighted)
	propbins0 = np.percentile(DATA, 100/nbins * np.arange(0, nbins+1, 1))
	if normcol[1] is not None:  # normalises data if norm column is given
		if normcol[0] == "dataset":
			DATA = dataset[:,col] - dataset[:,normcol[1]]
		elif normcol[0] == "data":
			DATA = data[:,1] - dataset[:,normcol[1]]
		else:
			raise NameError("Only keywords `dataset` and `data` accepted.")

	galprop = np.zeros_like(dataset[:,:2])  # columns are: HRS-id, property
	galprop[:,0] = dataset[:,0]
	# property bins (weighted)
	propbins = np.percentile(DATA, 100/nbins * np.arange(0, nbins+1, 1))
	galprop[:,1] = np.digitize(DATA, bins=propbins, right=True)  # put in bins
	galprop[:,1][galprop[:,1] == 0] = 1  # account for the open left interval

	# property cut
	T_prop = np.array([data[:,0][galprop[:,1]==i] for i in range(1,nbins+1)])
	M_prop = np.array([data[:,1][galprop[:,1]==i] for i in range(1,nbins+1)])

	return T_prop, M_prop, propbins, propbins0



def HistPlot(
		dataset, data, col, clrs, lbls, normcol=("which",None), nbins=3,
		framealpha=1, fontsize=10, lgd=(0,"upper right")):
	"""
	Plots histogram of galaxies according to a fiducial cut, specified by col.
	"""
	T_prop, M_prop, propbins, propbins0 = FiducialCut(dataset, data, col,
												normcol, nbins)
	fig, ax = plt.subplots(1, 2, figsize=(10,5))
	ax[0].set_ylabel(r"$\mathrm{Number \/\/ of \/\/ Galaxies}$", fontsize=20)
	ax[0].set_xlabel(r"$\mathrm{Dust \/ Temperature, \/ T/K}$", fontsize=20)
	ax[1].set_xlabel(r"$\mathrm{Dust \/ Mass, \/ \log \
					\left( M_d / M_{\odot} \right)}$", fontsize=20)

	[ax[i].hist(data[:,i], bins=20, color="grey", alpha=0.5,
				label=r"$\mathrm{HRS}$") for i in  range(2)]
	ax[0].hist(list(T_prop),  bins=20, histtype="step",
				lw=2, color=clrs, label=lbls)
	ax[1].hist(list(M_prop),  bins=20, histtype="step",
				lw=2, color=clrs, label=lbls)
	ax[lgd[0]].legend(loc=lgd[1], fancybox=True, framealpha=framealpha,
					fontsize=fontsize)



""" Imports """
parent = "Data"  # data parent folder
savedir = "Figures"  # figures2 save directory
saveout = "Output"  # data output save directory

# columns are: HRS-id, F100, e100, F160, e160 [Jy]
data1 = np.genfromtxt(parent + "/asu1.tsv", delimiter="\t", skip_header=55,
					usecols=((1,7,8,9,10)))
# columns are: S250 e250, S350, e350, S500, e500 [mJy]
data2 = np.genfromtxt(parent + "/asu2.tsv", delimiter="\t", skip_header=56,
					usecols=((2,3,4,5,6,7)))
# columns are: HRS-id, distance [Mpc]
data3 = np.loadtxt(parent + "/HRS_matt.txt", skiprows=1)
# !! Does not contain all galaxies !!
# columns are: HRS-id, log(M_dust) [Msol], dlog(M_dust), T [K], dT
data4 = np.genfromtxt(parent + "/asu3.tsv", delimiter="\t", skip_header=63,
					usecols=(0,5,6,2,3))
# columns are: HRS-id, Type, SFR (-/+), Ms (-/+), Z (-,+)
data5 = np.loadtxt(parent + "/HRS_metadata.csv", delimiter=",", skiprows=1)




""" Constants """
Msol = M_sun.value  # solar mass [kg]
Jy = 1e-26  # conversion factor from [Jy] to [SI]
f0 = 856.5e+9  # calibration reference frequency [Hz]
k0 = 0.192  # kappa_0 at f0 [m^2 / kg]
b_step = 0.05  # step of beta test values
beta = np.arange(0, 4+b_step, b_step)  # test beta values

max_iter = 3000  # max iterations of curve_fit function
nbins = 3  # number of histogram fiducial cuts
KS_pval = 0.05  # p-value of the Kolmogorov-Smirnov statistic



""" Data Manipulation """
# Merge Data
data1 = data1[data3[:,0].astype(int) - 1]
data2 = data2[data3[:,0].astype(int) - 1] * 1e-3  # [mJy] to [Jy]

data4[:,1:3] = 10**data4[:,1:3] * Msol  # converting dust mass to [kg]

data = np.column_stack((data1, data2, data3[:,-1]))  # data with merged columns
# HRS-id, F100, e100, F160, e160, S250, e250, S350, e350, S500, e500, D [Mpc]

# Disregard Undetected & Bad Galaxies; Accounting for Upper Limits
undetected = []  #  HRS-id's of undetected galaxies
uplim = [[[], []] for i in range(len(data))]  # flux lims: (HRS-id | lims)
for i in range(len(data)):  # loop over all galaxies
	uplim[i][0].append(data[:,0][i])  #  appends HRS-id
	if np.isnan(data[i][2:11:2]).any():  # check for NaN's in error
		temp = np.argwhere(np.isnan(data[i][2:11:2]))  # positions of NaN's
		for j in range(len(temp)):
			# identifies galaxies undetected in at least one wavelength
			# checks F-values: undetected gals have  upper F lim but NaN error
			# sets upper limits to NaN so they are not further accounted for
			if not np.isnan(data[i][2*temp[j,0]+1]):
				uplim[i][1].append(data[i][2*temp[j,0]+1])
				data[i][2*temp[j,0]+1] = np.nan
		# identifies galaxies undetected in at lease one wavelength
		if list(np.isnan(data[int(data[i,0])-1][2:11:2])).count(1) != 0:
			undetected.append(data[i,0])

badgal = [138, 183, 210, 228, 249]  # bad galaxies (228 not in the HRS volume)
# good galaxies
good = np.sort(np.array(list(set(data[:,0]) - set(undetected) - set(badgal))))
good = np.array([int(good[i]) for i in range(len(good))])  # as integers
data = data[good-1]  # disregards undetected galaxies
data5 = data5[good-1]  # disregards undetected galaxies
uplim = np.array(uplim)[good-1]  # disregards undetected galaxies

for i in range(len(uplim[:,1])):
	try:
		uplim[:,1] = Jy * uplim[:,1]  # upper limits in Jy
	except TypeError:
		continue
flux = Jy * data[:,1:11]  # flux/error in [SI] @ (100|160|250|350|500) um
# 5% calibration error
flux[:, 1::2] = np.sqrt((0.05 * flux[:, 1::2])**2 + flux[:, 1::2]**2)
wavl = np.array([100, 160, 250, 350, 500]) * 1e-6  # wavelength in [SI]
freq = c / wavl  # observation frequencies [SI]
# continuous array of frequencies [SI]
FREQ = np.logspace(np.log10(min(freq)), np.log10(max(freq)), 1000)



""" Data Analysis """
# Finding best-fit Parameters & Chi-squared Test (Bound beta)
# mean p_guess [kg, K]
p_guess = [np.mean(data4[:,i][np.isnan(data4[:,i]) == False]) for i in [1, 3]]

# initializing chi-sq and d.o.f. arrays
chi2, prob = [np.zeros((len(beta), len(data))) for i in range(2)]
popt = np.zeros((len(beta), len(data), 2))  # initializing popt array
pcov = np.zeros((len(beta), len(data), 2, 2))  # initializing pcov array
CHI2 = np.zeros((len(beta), 2))  # median & mean chi-squared
for j, k in enumerate(beta):  # loop over all values of beta
	for i in range(len(data)):  # loop over all galaxies
		# positions of unmeasured fluxes
		nonan = np.where(np.isnan(flux[i][::2]) == False)
		D = data[i,11]  # distance of i-th galaxy in [pc]

		# if popt does not exist or is NaN
		if (data4[:,0].any() != data[i,0]) or \
			(np.isnan(data4[np.where(data4[:,0] == data[i,0])]).any()):
			# optimal parameters
			popt[j,i], pcov[j,i] = curve_fit(lambda freq, M, T:
										PlanckMod(freq, D, M, T, beta=k),
										freq[nonan], flux[i][::2][nonan],
										sigma=flux[i][1::2][nonan],
										p0=p_guess, maxfev=max_iter)
		else:
			# optimal parameters
			popt[j,i], pcov[j,i] = curve_fit(lambda freq, M, T:
										PlanckMod(freq, D, M, T, beta=k),
										freq[nonan], flux[i][::2][nonan],
										sigma=flux[i][1::2][nonan],
										p0=data4[i,1::2], maxfev=max_iter)

		# Chi-squared Statistic
		# model flux values
		flux_chi2 = PlanckMod(freq[nonan], data[i,11], *popt[j,i], beta=k)
		# reduced chi-sq
		chi2[j,i] = sum(((flux_chi2 - flux[i][::2][nonan])
						/ flux[i][1::2][nonan])**2) / len(nonan[0])
		# probability of chi-sq with (no. of datapoints) d.o.f.
		prob[j,i] = chisq.cdf(len(nonan[0]) * chi2[j,i], len(nonan[0]))

	# median chi-sq value
	CHI2[j,0] = np.median(chi2[j][np.where(np.isnan(chi2[j]) == False)])
	# mean chi-sq value
	CHI2[j,1] = np.mean(chi2[j][np.where(np.isnan(chi2[j]) == False)])


# Best-fit parameters
bopt = np.argmin(CHI2[:,0])  # arg of optimal beta (based on median value)
M, T = popt[bopt].T  # mass & temperature fits
# fit uncertainties
dM,dT = np.array([np.sqrt(np.diag(pcov[bopt,i])) for i in range(len(data))]).T
logM = np.log10(M / Msol)  # log mass in solar units
alldata = np.column_stack((T, logM))  # all data (used in HistPlot method)
# Exporting best-fit Parameters
header = "HRS-id,log(M/Msol),log(dM/Msol),T/K,dT/K,chi-sq,prob"
X = np.column_stack((data[:,0], logM, np.log10(dM / Msol), T, dT, chi2[bopt],
					prob[bopt]))
np.savetxt(saveout + "/fit_params.csv", X, fmt="%.5f", delimiter=",",
			header=header)

# Finding best-fit Parameters & Chi-squared Test (Free beta)
# initializing chi-sq and d.o.f. arrays
chi2_beta, prob_beta = [np.zeros_like(data[:,0]) for i in range(2)]
popt_beta = np.zeros((len(data), 3))  # initializing popt array
pcov_beta = np.zeros((len(data), 3, 3))  # initializing pcov array
for i in range(len(data)):  # loop over all galaxies
	# positions of unmeasured fluxes
	nonan = np.where(np.isnan(flux[i][1::2]) == False)
	D = data[i,11]  # distance of i-th galaxy in [pc]

	# if popt does not exist or is NaN
	if (data4[:,0].any() != data[i,0]) or \
		(np.isnan(data4[np.where(data4[:,0] == data[i,0])]).any()):
		# optimal parameters
		popt_beta[i], pcov_beta[i] = curve_fit(lambda freq, M, T, beta:
										PlanckMod(freq, D, M, T, beta),
										freq[nonan], flux[i][::2][nonan],
										sigma=flux[i][1::2][nonan],
										p0=np.append(np.array(p_guess),
										beta[bopt]), maxfev=max_iter)
	else:
		# optimal parameters
		popt_beta[i], pcov_beta[i] = curve_fit(lambda freq, M, T, beta:
										PlanckMod(freq, D, M, T, beta),
										freq[nonan], flux[i][::2][nonan],
										sigma=flux[i][1::2][nonan],
										p0=np.append(data4[i,1::2],
										beta[bopt]), maxfev=max_iter)

	# Chi-squared Statistic
	# model flux values
	flux_chi2_beta = PlanckMod(freq[nonan], data[i,11], *popt_beta[i])
	# reduced chi-sq
	chi2_beta[i] = sum(((flux_chi2_beta - flux[i][::2][nonan])
					/ flux[i][1::2][nonan])**2) / len(nonan[0])
	# probability of chi-sq with (no. of datapoints) d.o.f.
	prob_beta[i] = chisq.cdf(len(nonan[0]) * chi2_beta[i], len(nonan[0]))


# Best-fit parameters
M_beta, T_beta, beta_beta = popt_beta.T  # mass & temperature fits
# fit uncertainties
dM_beta, dT_beta, db_beta = np.array([np.sqrt(np.diag(pcov_beta[i]))
									for i in range(len(data))]).T
logM_beta = np.log10(M_beta / Msol)  # log mass in solar units
alldata_beta = np.column_stack((T_beta, logM_beta))  # all data
# Exporting best-fit Parameters
header = "HRS-id,log(M/Msol),log(dM/Msol),T/K,dT/K,chi-sq,prob"
X = np.column_stack((data[:,0], logM_beta, np.log10(dM_beta / Msol), T_beta,
					dT_beta, chi2_beta, prob_beta))
np.savetxt(saveout + "/fit_params_beta.csv", X, fmt="%.5f", delimiter=",",
			header=header)

# Galaxy type cut
ellipticals = np.arange(-3, 3, 1)  # elliptical galaxy types
spirals = np.append(np.arange(3, 12, 1), 18)  # spiral galaxy types
irregulars = np.append(np.arange(12, 18, 1), 19)

galtype = np.zeros((len(data),2), dtype=float)  # columns are: HRS-id, gal type
galtype[:,0] = data5[:,0]
for i in range(len(data)):
	if int(data5[i,1]) in ellipticals:
		galtype[i,1] = 1  # flag "1" for ellipticals
	elif int(data5[i,1]) in spirals:
		galtype[i,1] = 2  # flag "2" for spirals
	elif int(data5[i,1]) in irregulars:
		galtype[i,1] = 3  # flag "3" for irregulars/peculiars
	else:
		print("galaxy type not listed")

T_type = np.array([T[galtype[:,1]==i] for i in range(1,4)])  # gal type cut
M_type = np.array([logM[galtype[:,1]==i] for i in range(1,4)])  # gal type cut


# Kolmogorov-Smirnov (K-S) Statistic
T_mass, M_mass, massbins, massbins0 = FiducialCut(data5, alldata, col=5,
											normcol=("data",5))  # mass cut
T_sfr, M_sfr, sfrbins, sfrbins0 = FiducialCut(data5, alldata, col=2,
											normcol=("dataset",5))  # sfr cut
T_met, M_met, metbins, _ = FiducialCut(data5, alldata, col=8)  # metallicity cut

KS_Ttype = min([ks_2samp(T_type[i], T_type[(i+1) % nbins])[1] for i in range(nbins)])
KS_Tmass = min([ks_2samp(T_mass[i], T_mass[(i+1) % nbins])[1] for i in range(nbins)])
KS_Tsfr = min([ks_2samp(T_sfr[i], T_sfr[(i+1) % nbins])[1] for i in range(nbins)])
KS_Tmet = min([ks_2samp(T_met[i], T_met[(i+1) % nbins])[1] for i in range(nbins)])

KS_Mtype = min([ks_2samp(M_type[i], M_type[(i+1) % nbins])[1] for i in range(nbins)])
KS_Mmass = min([ks_2samp(M_mass[i], M_mass[(i+1) % nbins])[1] for i in range(nbins)])
KS_Msfr = min([ks_2samp(M_sfr[i], M_sfr[(i+1) % nbins])[1] for i in range(nbins)])
KS_Mmet = min([ks_2samp(M_met[i], M_met[(i+1) % nbins])[1] for i in range(nbins)])
# Exporting best-fit Parameters
header = "T_type,T_mass,T_sfr,T_met,M_type,M_mass,M_sfr,M_met"
X = np.column_stack((
		KS_Ttype, KS_Tmass, KS_Tsfr, KS_Tmet,
		KS_Mtype, KS_Mmass, KS_Msfr, KS_Mmet
		))
np.savetxt(saveout + "/KS_stats.csv", X, fmt="%.3e", delimiter=",",
			header=header)
X_bins = np.array([massbins0, sfrbins0, metbins])  # all bins
# bin midpoints
midbins = np.array([(X_bins[i][:-1]+X_bins[i][1:])/2 for i in range(len(X_bins))])
# combines KS stats and discards type distinction
X_KS = np.hstack((np.delete(X, [0, 4], axis=1)))
X_cut = [T_mass,T_sfr,T_met,M_mass,M_sfr,M_met]  # all data

# initialising mean and std arrays
mn, sd = [np.zeros((len(X_cut), nbins)) for i in range(2)]
# these are 2D arrays with (nrows)==(number of KS stats) and (ncols)==(nbins)
prints = np.append(np.array(header.split(","))[1:len(X_bins)+1],
							np.array(header.split(","))[len(X_bins)+2:])

depend = np.array([])
print("\nQuantities with KS statistic < %.3f:" % KS_pval)
# if one of the two (dust mass, dust temperature) is bin-dependent
# then both are assumed dependent
for i in range(len(X_cut)):
	# mean and standard deviation
	mn[i] = [np.mean(X_cut[i][j]) for j in range(nbins)]
	sd[i] = [np.std(X_cut[i][j]) / np.sqrt(len(X_cut[i][j])) for j in range(nbins)]
	if X_KS[i] < KS_pval:  # p-value of the KS-statistic
		# prints quantity which is property dependent
		print(prints[i] + "\t%.6f" % X_KS[i])
		depend = np.append(depend, [i, (i+len(X_bins))%len(X_bins)])
depend = np.sort(list(set(depend))).astype(int)  # removes duplicates and sorts

X_bins = X_bins[depend[:len(depend)/2]]  # only keeps dependent
midbins = midbins[depend[:len(depend)/2]] # only keeps dependent
mn = mn[depend]  # only keeps dependent
sd = sd[depend]  # only keeps dependent

mn[len(mn)/2:] -= midbins[0]  # specific dust mass



""" Plots """
## Best-fit Plots
ncol = 4  # number of columns in the plot
nrow = 10  # number of rows in the plot
low, high = 0.95, 1.05  # axes limits

temp = 1  # temporary variable for file saving
for i, j in enumerate(data[:,0]):
	# remaining galaxies (-1 +1 because it is not plotted yet)
	remaining = len(data) - i
	D = data[i,11]  # distance of i-th galaxy in [pc]

	FLUX = PlanckMod(FREQ, D, *popt[bopt,i], beta=beta[bopt])  # flux fit
	# 1-sigma error in flux fit
	u = h * FREQ / (kB * T[i])  # variable change
	dF = np.sqrt(FLUX**2 * (dM[i] / M[i])**2 +
				(FLUX * np.exp(u) * u / (np.exp(u)-1))**2 * (dT[i] / T[i])**2)
	Fmin, Fmax = FLUX - dF, FLUX + dF  # 1-sigma extrema

	if i % (nrow * ncol) == 0:  # initializing plot
		if remaining <= (nrow-1) * ncol:
			# new number of rows for final figure
			remrow = np.ceil(float(remaining ) / ncol)
			fig, ax = plt.subplots(int(remrow), ncol, sharex=True,
								figsize=(2.5*ncol,3*remrow))
		else:
			fig, ax = plt.subplots(int(nrow), ncol, sharex=True,
								figsize=(2.5*ncol,3*nrow))

		fig.tight_layout(w_pad=0)
		fig.subplots_adjust(hspace=0)  # join adjacent x-axes
		fig.text(0.5, -0.02, r"Wavelength, $\lambda / \mathrm{\mu m}$",
				ha="center", fontsize=14)  # xlabel
		fig.text(-0.03, 0.5, r"Flux, $F_{\lambda} / \mathrm{Jy}$",
				va="center", rotation="vertical", fontsize=14)  # ylabel

	row = int(i / ncol % nrow)  # row count restarts for each new figure
	col = i % ncol  # column count restarts in every new line

	# Formatting
	ax[row,col].set_xscale("log")
	ax[row,col].set_yscale("log")
	ax[row,col].grid(which="both", ls=":")
	ax[row,col].set_xlim(low * 1e+3 * min(c / FREQ), high * 1e+3 * max(c / FREQ))
	ax[row,col].set_ylim(low * min(np.append(Fmin, flux[i][::2] - flux[i][1::2]) / Jy),
				high * max(np.append(Fmax, flux[i][::2] + flux[i][1::2]) / Jy))
	if j == 243: ax[row,col].set_ylim(0.3,)  # ylim exception
	# disable minor y tick labels
	ax[row,col].yaxis.set_minor_formatter(FormatStrFormatter(""))
	# format y axis tick labels
	ax[row,col].yaxis.set_major_formatter(FuncFormatter(lambda y, pos:
									('{{:.{:1d}f}}'.format(int(np.maximum(
									-np.log10(np.abs(y)), 0)))).format(y)))
	# legend
	ax[row,col].annotate(
		"$%d$" % j, xy=(.97,.92), xycoords="axes fraction",
		size=10, ha="right", va="top", bbox=dict(boxstyle="round", fc="w"))

	# Plotting
	# 1-sigma range
	ax[row,col].fill_between(1e+3 * c / FREQ, Fmax / Jy, Fmin / Jy,
							color="lightgreen")
	ax[row,col].plot(1e+3 * c / FREQ, FLUX / Jy, "r-", lw=1.5)  # best fit
	# data points
	ax[row,col].errorbar(1e+3 * c / freq, flux[i][::2] / Jy, yerr=flux[i][1::2] / Jy,
					 fmt="bo", ecolor="k", elinewidth=2, label="$%d$" %j)

	if (row == nrow-1) or (remaining <= ncol):
		# disable tick labels
		ax[row,col].xaxis.set_minor_formatter(FormatStrFormatter(""))
		# custom x tick labels
		plt.xticks(1e+3 * c / freq, [100, 160, 250, 350, 500])
		# finalising the figure
		if ((row + 1) * (col + 1) == nrow * ncol) or i == len(data) - 1:
			# switches off empty subplots (only last row)
			if i == len(data) - 1:
				remrow = min(nrow, remrow)
				[ax[-1,-num].axis("off") for num in
					range(1, int(remrow*ncol - (row*ncol + (col+1))+1))]
			# save and close figure if (i) last subplot is filled or
			# (ii) the end of the loop is reached
			fig.savefig(savedir + "/galfit_%d.png" % temp, dpi=200,
						bbox_inches="tight")
			plt.close()
			temp += 1  # temp var for filenaming on saving


# Chi-squared Plot
x1, x2, y1, y2 = 1.4, 2.3, 0.4, 1.5  # x lims, y lims

fig, ax = plt.subplots(1,1)
ax.grid("on", ls=":")
ax.set_xlim(min(beta), max(beta))
#ax.set_ylim(y1,)
ax.set_xlabel(r"$\beta$", fontsize=20)
ax.set_ylabel(r"$\mathrm{Reduced} \/ \chi^2$", fontsize=20)
ax.set_yscale("log")
# format y axis tick labels
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos:	('{{:.{:1d}f}}'.format(
						int(np.maximum(-np.log10(np.abs(y)), 0)))).format(y)))


color = "navy"
ax.plot(beta, CHI2[:,0], ls="-", c=color, lw=2,
		label=r"$\widetilde{ \chi^2 } / N $")
ax.plot(beta, CHI2[:,1], ls="--", c=color, lw=2,
		label=r"$\widebar{ \chi^2 } / N $")
# Zoomed subplot
axins = plt.axes([.61, .16, .25, .25])
twinax = axins.twinx()
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
twinax.set_xlim(x1, x2)
axins.set_xticks(np.arange(x1, x2, 0.2))
axins.tick_params(axis='both', which='major', labelsize=10)
twinax.tick_params(axis='both', which='major', labelsize=10)
axins.grid("on", ls=":")
mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.7")

twinax.hist(popt_beta[:,-1], bins=35, alpha=0.5, color="gray", bottom=y1)
axins.plot(beta, CHI2[:,0], ls="-", c=color, lw=2,
			label=r"$\widetilde{ \chi_2^2 } / N $")
axins.plot(beta, CHI2[:,1], ls="--", c=color, lw=2,
			label=r"$\widebar{ \chi_2^2 } / N $")
leg = ax.legend(loc="upper right", fontsize=14, fancybox=True)
leg.get_frame().set_alpha(1)
fig.savefig(savedir + "/chisq.png", dpi=300, bbox_inches="tight")
plt.close()


# Distribution of Chi-square (Bound beta)
fig, ax = plt.subplots(1,1)
ax.set_xlim(0, max(chi2[bopt]))
ax.set_xlabel(r"$\mathrm{\chi^2 / N}$", fontsize=20)
ax.set_ylabel(r"$\mathrm{Number \/\/ of \/\/ Galaxies}$", fontsize=20)
bins = np.linspace(min(chi2[bopt]), max(chi2[bopt]), 20+1)
x_chimodel = (bins[1:] + bins[:-1]) / 2  # midpoint of each bin
y_chimodel = len(data) * chisq.pdf(x_chimodel, df=1)  # model chi squared of df=1

H, _ = np.histogram(chi2[bopt], bins=bins)
ax.bar((
	bins[1:]+bins[:-1])/2, H, yerr=np.sqrt(H),
	width=bins[1]-bins[0], alpha=0.5, color="gray", label=r"$\mathrm{Data}$"
	)
ax.plot(x_chimodel, y_chimodel, "r-", lw=2, label=r"$\chi^2_1 \/ \mathrm{fit}$")
ax.plot(x_chimodel, y_chimodel, "ko", ms=5)
ax.legend(loc="upper right", fontsize=14)
fig.savefig(savedir + "/chidist.png", dpi=300, bbox_inches="tight")
plt.close()


# Temperature and Dust Mass Histograms
# galaxy type
fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].set_ylabel(r"$\mathrm{Number \/\/ of \/\/ Galaxies}$", fontsize=20)
ax[0].set_xlabel(r"$\mathrm{Dust \/ Temperature, \/ T / K}$", fontsize=20)
ax[1].set_xlabel(r"$\mathrm{Dust \/ Mass, \/ \log \
				\left( M_d / M_{\odot} \right)}$", fontsize=20)
colors = ["darkorange", "royalblue", "tomato"]
labels = [r"$\mathrm{E/S0/dE}$", r"$\mathrm{S/Sb}$", r"$\mathrm{Irr/Pec}$"]

ax[0].hist(T, bins=20, color="grey", alpha=0.5, label=r"$\mathrm{HRS}$")
ax[1].hist(np.log10(M / Msol), bins=20,
			color="grey", alpha=0.5, label=r"$\mathrm{HRS}$")
ax[0].hist(
	list(T_type), bins=20, histtype="step",
	rwidth=1, lw=2, color=colors, label=labels
	)
ax[1].hist(
	list(M_type), bins=20, histtype="step",
	rwidth=1, lw=2, color=colors, label=labels)
ax[0].legend(loc="upper right", fancybox=True)
fig.savefig(savedir + "/galhist_type.png", dpi=300, bbox_inches="tight")
plt.close()

# stellar mass
colors = ["lightcoral", "red", "darkred"]
labels = ["$\\log{\\left( \\frac{M_d}{M_s} \\right)} < %.2f$" % massbins[1],
		"$%.2f \\leq \\log{\\left( \\frac{M_d}{M_s} \\right)} < %.2f$"
		% (massbins[1], massbins[2]),
		"$\\log{\\left( \\frac{M_d}{M_s} \\right)} \\geq %.2f$" % massbins[2]]
HistPlot(
	data5, alldata, col=5, normcol=("data", 5), nbins=nbins,
	clrs=colors, lbls=labels, lgd=(1,"upper left"), framealpha=0, fontsize=8
	)
plt.savefig(savedir + "/galhist_mass.png", dpi=300, bbox_inches="tight")
plt.close()

# star formation rate
colors = ["c", "royalblue", "midnightblue"]
labels = ["$\\log{ \\mathrm{ \\frac{SFR}{M_s \/ yr} } } < %.2f$" % sfrbins[1],
		"$%.2f \\leq \\log{\\mathrm{ \\frac{SFR}{M_s \/ yr} }} < %.2f$"
		% (sfrbins[1], sfrbins[2]),
		"$\\log{\\mathrm{ \\frac{SFR}{M_s \/ yr} }} \\geq %.2f$" % sfrbins[2]]
HistPlot(
	data5, alldata, col=5, normcol=("dataset",2), nbins=nbins,
	clrs=colors, lbls=labels, lgd=(1,"upper left"), framealpha=0, fontsize=7
	)
plt.savefig(savedir + "/galhist_sfr.png", dpi=300, bbox_inches="tight")
plt.close()

# metallicity
colors = ["gold", "darkorange", "chocolate"]
labels = ["$\\mathrm{Z} < %.2f$" % metbins[1],
		"$%.2f \\leq \\mathrm{Z} < %.2f$" % (metbins[1], metbins[2]),
		"$\\mathrm{Z} \\geq %.2f$" % metbins[2]]
HistPlot(
	data5, alldata, col=8, nbins=nbins, clrs=colors, lbls=labels,
	lgd=(1,"upper left"), framealpha=0, fontsize=10
	)
plt.savefig(savedir + "/galhist_met.png", dpi=300, bbox_inches="tight")
plt.close()


# Distribution of Optimal beta for each Galaxy (Free beta)
fig, ax = plt.subplots(1,1)
ax.set_xlabel(r"$\beta$", fontsize=20)
ax.set_ylabel(r"$\mathrm{Number \/\/ of \/\/ Galaxies}$", fontsize=20)

colors = ["darkorange", "royalblue", "tomato"]
labels = [r"$\mathrm{E/S0/dE}$", r"$\mathrm{S/Sb}$", r"$\mathrm{Irr/Pec}$"]

H = ax.hist(beta_beta, bins=25, histtype="stepfilled", color="gray", alpha=0.5,
											lw=2, label=r"$\mathrm{HRS}$")
[ax.hist(beta_beta[galtype[:,1] == i+1], bins=H[1], histtype="step", lw=2,
		color=colors[i], label=labels[i]) for i in range(3)]
ax.plot(np.mean(beta_beta) * np.ones(2), np.linspace(0, max(H[0]), 2),
		"r--", label=r"$\mathrm{mean}$")
ax.plot(np.median(beta_beta) * np.ones(2), np.linspace(0, max(H[0]), 2),
		"--", c="orange", label=r"$\mathrm{median}$")
ax.legend(loc="upper right", fontsize=14)
fig.savefig(savedir + "/bopt.png", dpi=300, bbox_inches="tight")
plt.close()


# Dependency on Fiducial Cuts
x_labels = [r"$\mathrm{Stellar \/ Mass, \/ \log \left( M_s / M_{\odot} \right)}$",
			r"$\mathrm{Metallicity, \/\/ \mathit{Z}/Z_{\odot}}$"]
fig, ax = plt.subplots(2, len(X_bins), sharex="col", sharey="row")
ax = np.ndarray.flatten(ax)  # flatten Axis object
ax[0].set_ylabel(r"$\mathrm{Dust \/ Temperature, \/ T / K}$", fontsize=10)
ax[len(X_bins)].set_ylabel(r"$\mathrm{Specific \/ Dust \/ Mass, \/ \
						\log \left( M_d / M_s \right)}$", fontsize=10)
for i in range(len(X_bins)):
	j = i+len(X_bins)  # second graph
	# plots bin edges
	for k in range(len(X_bins)):
		temp = np.ones((2, len(X_bins[i][1:-1])))
		vmin1 = np.amin(mn[:len(mn)/2] - sd[:len(mn)/2])
		vmax1 = np.amax(mn[:len(mn)/2] + sd[:len(mn)/2])
		vmin2 = np.amin(mn[len(mn)/2:] - sd[len(mn)/2:])
		vmax2 = np.amax(mn[len(mn)/2:] + sd[len(mn)/2:])

		ax[i].plot(X_bins[i][1:-1][k] * temp[:,k], np.linspace(vmin1, vmax1, 2),
					"-.", c="tomato")
		ax[j].plot(X_bins[i][1:-1][k] * temp[:,k], np.linspace(vmin2, vmax2, 2),
					"-.", c="tomato")

	ax[i].plot(midbins[i], mn[i], "k--")
	ax[i].errorbar(midbins[i], mn[i], yerr=sd[i], fmt="o",
				color="royalblue", capsize=2)

	ax[j].plot(midbins[i], mn[j], "k--")
	ax[j].errorbar(midbins[i], mn[j], yerr=sd[j], fmt="o",
				color="royalblue", capsize=2)
	ax[j].set_xlabel(x_labels[i], fontsize=10)

	ax[i].grid("on", ls=":", alpha=0.4)
	ax[j].grid("on", ls=":", alpha=0.4)
plt.tight_layout()
fig.savefig(savedir + "/dependencies.png", dpi=300, bbox_inches="tight")
plt.close()


# Ms - Z Cross Correlation
colors = ["darkorange", "royalblue", "tomato"]
labels = [r"$\mathrm{E/S0/dE}$", r"$\mathrm{S/Sb}$", r"$\mathrm{Irr/Pec}$"]
fig, ax = plt.subplots(1,1)
ax.grid("on", ls=":")
ax.set_xlabel(r"$\mathrm{Stellar \/ Mass, \/ \log \
			\left( M_s / M_{\odot} \right)}$", fontsize=20)
ax.set_ylabel(r"$\mathrm{Metallicity, \/\/ \mathit{Z}/Z_{\odot}}$", fontsize=20)
_ = [ax.plot(data5[:,5][galtype[:,1] == i+1], data5[:,8][galtype[:,1] == i+1],
			"o", color=colors[i], ms=5, label=labels[i]) for i in range(3)]
ax.legend(loc="upper left", fontsize=12, framealpha=1)
plt.tight_layout()
fig.savefig(savedir + "/cross_corr.png", dpi=300, bbox_inchess="tight")
plt.close()


# Specific Dust Mass - Stellar Mass (type distinction)
# run KS stats section with redefined arrays as shown below
# after each run, redifine the whole sample from the relevant lines of code
# each galaxy type is denoted by numbers {1,2,3}
# -----------------------------------------------------------------------------
# data5 = data5[galtype[:,1] == 1]  # selects galaxy type 1
# alldata = alldata[galtype[:,1] == 1]  # selects galaxy type 1
# X = X[galtype[:,1] == 1]  # selects galaxy type 1
# -----------------------------------------------------------------------------
# data is saved using code shown below; it is then loaded to plot the graph
# care which mn and sd line is extracted as this changes for each subsample
# Egals = np.column_stack((midbins[0], mn[2], sd[2]))  # E for ellipticals
# np.save(saveout + "/Egals", Egals)  # saves a .npy array file for E type gals

Agals = np.load(saveout + "/allgals.npy")  # all galaxies
Egals = np.load(saveout + "/Egals.npy")  # ellipticals
Sgals = np.load(saveout + "/Sgals.npy")  # spirals
Igals = np.load(saveout + "/Igals.npy")  # irregulars
# all data; like data in same line (transposed)
dists = [Agals.T, Egals.T, Sgals.T, Igals.T]

colors = ["grey", "darkorange", "royalblue", "tomato"]
labels = [
		r"$\mathrm{HRS}$",
		r"$\mathrm{E/S0/dE}$",
		r"$\mathrm{S/Sb}$",
		r"$\mathrm{Irr/Pec}$"
		]

fig, ax = plt.subplots(1)
ax.set_xlabel(r"$\mathrm{Stellar \/ Mass, \/ \
			\log \left( M_s / M_{\odot} \right)}$", fontsize=16)
ax.set_ylabel(r"$\mathrm{Specific \/ Dust \/ Mass, \/ \
			\log \left( M_d / M_s \right)}$", fontsize=16)

for i in range(len(dists)):
	ax.plot(dists[i][0], dists[i][1], "--", c=colors[i])
	ax.errorbar(dists[i][0], dists[i][1], yerr=dists[i][2], fmt="o",
						color=colors[i], capsize=2, label=labels[i])

ax.grid("on", ls=":", alpha=0.4)
ax.legend(loc="upper right", fontsize=12)
plt.tight_layout()
fig.savefig(savedir + "/dustmass_galtype.png", dpi=300, bbox_inches="tight")
plt.close()
