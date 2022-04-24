import fulmar
from wotan import flatten, transit_mask
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.lines as lines
from astrobase import lcmath
from astrobase.periodbase import kbls
import logging, os


logging.basicConfig(filename='tool.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Cleam Time Gap will clean zero value time gaps greater than gapLength
# and insert a separator
# 
# Inputs:	
#
# array (Numpy Array) time vector for which you want to clean time gaps
# 
# gapLenght (int) the minimun size a gap must have to be clean
#
# separator (int) the minimun separation that must be left between time groups
#
# Output:
#
# array_new (Numpy Array) a new array without time gaps

def CleanTimeGap(array, gapLenght, separator):     

	n_tgroups, indexes = lcmath.find_lc_timegroups(array, mingap=gapLenght)
	timestamps_end = []
	timestamps_in = []
	marks = []
	diff = 0

	for i in range(n_tgroups):
		timestamps_in.append(array[indexes[i]][0])
		timestamps_end.append(array[indexes[i]][len(array[indexes[i]])-1])
		
	for i in range(n_tgroups):
		if i == 0:
			array_new = array[indexes[0]]
		else:
			diff = (timestamps_in[i] - timestamps_end[i-1]) - separator + diff
			marks.append(array_new[len(array_new)-1] + separator/2)
			array_new = np.append(array_new, array[indexes[i]] - diff)

	return array_new, marks

# Function to compile the different PNG temporary files into one PDF report
def PDFCompiler(tic_id, SDEmax, search):

	image_list = []

	if os.path.exists('im_0.png'):
		image_1 = Image.open(r'im_0.png')
		im_1 = image_1.convert('RGB')
	if os.path.exists('im_1.png'):
		image_2 = Image.open(r'im_1.png')
		im_2 = image_2.convert('RGB')
		image_list.append(im_2)
	if os.path.exists('im_2.png'):
		image_3 = Image.open(r'im_2.png')
		im_3 = image_3.convert('RGB')
		image_list.append(im_3)
	if os.path.exists('im_3.png'):
		image_4 = Image.open(r'im_3.png')
		im_4 = image_4.convert('RGB')
		image_list.append(im_4)
	if os.path.exists('im_4.png'):
		image_5 = Image.open(r'im_4.png')
		im_5 = image_5.convert('RGB')
		image_list.append(im_5)

	if os.path.exists('im_0.png'):
		im_1.save(r'Reports/Exposure Time 120s_ Last Run/' + str(round(SDEmax,2)) + '_search_Nº'+ str(search+1) +'_'+tic_id+'.pdf', save_all=True, append_images=image_list)


# Target name input / multiple targets
# Open txt with the list of candidates
tgts = []
expt = 120

f = open("tgts.txt", "r")
for x in f:
	tgts.append(x)


for tgt in tgts:

	SDEmax = 0
	search = 0

	try:
	# Target lookup and creation
		lc_targ = fulmar.target(tgt, mission='TESS')

	# Building the lightcurve
		lc_targ.build_lightcurve(author='SPOC', exptime=expt) #SPOC FOR TESS
	
	except Exception as e:
		print(e, file=open("log.txt", "a"))
		print(tgt,'NOT LOADED -- WARNING\n', file=open("log.txt", "a"))
		continue

	# Mask general outliers using Fulmar  
	m1 = lc_targ.mask_outliers(sigma=7)
	lc_targ.ts_stitch = lc_targ.ts_stitch[m1]
	
	# Clean NaN values from the arrays
	lc_time = lc_targ.ts_stitch.time.value[~np.isnan(lc_targ.ts_stitch['flux'])]
	lc_err = lc_targ.ts_stitch['flux_err'][~np.isnan(lc_targ.ts_stitch['flux'])]
	flatten_lc1 = lc_targ.ts_stitch['flux'][~np.isnan(lc_targ.ts_stitch['flux'])]	

	# Sigma clipping with flux error
	err_std = np.std(lc_err)
	err_mean = np.mean(lc_err)
	sig = 3

	lc_time = lc_time[np.where(np.logical_and(lc_err < err_mean + sig*err_std, lc_err > err_mean - sig*err_std))]
	flatten_lc1 = flatten_lc1[np.where(np.logical_and(lc_err < err_mean + sig*err_std, lc_err > err_mean - sig*err_std))]
	lc_err = lc_err[np.where(np.logical_and(lc_err < err_mean + sig*err_std, lc_err > err_mean - sig*err_std))]

	# Wotan detrending with biweight method
	flatten_lc1, trend = flatten(lc_time, flatten_lc1,
		method='biweight', window_length=0.75, edge_cutoff=0.5,
		break_tolerance=0.1, return_trend=True)
	#flatten_lc1 = sigma_clip(flatten_lc1, sigma_upper=3, sigma_lower=20)

	# Time normalization to zero
	if len(lc_time) > 0:
		zero_time = lc_time[0]
		lc_time = lc_time - zero_time
	else:
		print(tgt, 'Lightcurve empty', file=open("log.txt", "a"))
		continue

	lc_time = lc_time[~np.isnan(flatten_lc1)]
	lc_err = lc_err[~np.isnan(flatten_lc1)]
	flatten_lc1 = flatten_lc1[~np.isnan(flatten_lc1)]	

	# Masking bad quality LCs
	# maskara = np.where(np.logical_and(lc_time>25, lc_time<50))
	# maskara = np.where(lc_time<50)
	# lc_time = lc_time[maskara]
	# lc_err = lc_err[maskara]
	# flatten_lc1 = flatten_lc1[maskara]


	# Searching for at least three planets
	for pt in range(5):
			
		# Using BLS parallel method to rapid detection
		bls_results = kbls.bls_parallel_pfind(lc_time, flatten_lc1,
			lc_err, startp=0.179, endp=100, magsarefluxes=True,
			nbestpeaks=5, get_stats=True, sigclip=10, mintransitduration= 0.01)
			

		# NOTE: the combined BLS spectrum produced by this function is not identical to that produced 
		# by running BLS in one shot for the entire frequency space. There are differences on the order 
		# of 1.0e-3 or so in the respective peak values, but peaks appear at the same frequencies for 
		# both methods. This is likely due to different aliasing caused by smaller chunks of the 
		# frequency space used by the parallel workers in this function. When in doubt, confirm results 
		# for this parallel implementation by comparing to those from the serial implementation below.

		#bls_results = kbls.bls_serial_pfind(lc_time, flatten_lc1, lc_err, startp=0.32, endp=100, magsarefluxes=True, nbestpeaks=5, get_stats=True, sigclip=[10.,5.], mintransitduration= 0.01 , maxtransitduration=0.10)
		
		# Masking NaNs and Infs
		if bls_results['lspvals'] is not None:
			power = bls_results['lspvals'][np.isfinite(bls_results['lspvals'])]
			periods = bls_results['periods'][np.isfinite(bls_results['lspvals'])]
		else:
			print(tgt,' had no power spectrum\n', file=open("log.txt", "a"))
			break

		fig = plt.figure(figsize = (20,10) )
	
		# Easy Handles
		stats = bls_results['stats'][0]
		epoch0 = stats['epoch']
		tdur = stats['transitduration'] #Transit Duration in phase units
		p = stats['period'] # Period in days
		bls = stats['blsmodel'] # BLS model values
		phases = stats['phases'] # Phases values
		depth = stats['transitdepth'] # Transit depth

		# Calculate SDE 
		SDE = (power[np.where(periods == bls_results['bestperiod'])][0] - np.mean(power)) / np.std(power)
		
		# Saving max SDE and searchc number
		if SDE > SDEmax:
			SDEmax = SDE
			search = pt

		# Model wrap for plotting
		bls = np.concatenate((bls[np.where(phases>0.8)[0]],bls[np.where(np.logical_and(phases<=0.8, phases>=0))[0]]))			
		phases[np.where(phases>0.8)[0]] -= 1
		phases = np.sort(phases)

		# Astrobase methods

		# Phasing lightcurve
		phased_results = lcmath.phase_magseries_with_errs(lc_time, flatten_lc1, lc_err, bls_results['bestperiod'], epoch0, wrap=True, sort=True)

		# Binning phased lightcurve
		# Bin zise is 1/10 transit duration
		if tdur*0.1	< 0.0005:
			bindur = 0.01
		else:
			bindur	= tdur
		binned_results = lcmath.phase_bin_magseries(phased_results['phase'], phased_results['mags'], binsize=bindur*0.1, minbinelems=7)
		
		# Intransit Masking
		intransit = transit_mask(lc_time, bls_results['bestperiod'], tdur*bls_results['bestperiod'], epoch0)

		# Testing depth 
		if depth > 0:

			# Plotting detrended lightcurve and marking transit~
			ax0 = plt.subplot2grid((2,4),(0,0),colspan=2)
			ax0.set_title('Detrended Lightcurve')
			lc_time_clean, marks = CleanTimeGap(lc_time, 30, 5)
			ax0.plot(lc_time_clean[~intransit], flatten_lc1[~intransit],'k.', markersize=1.8, alpha = 0.25)
			ax0.plot(lc_time_clean[intransit], flatten_lc1[intransit],'o', color='orange', markersize=2, alpha = 0.5, label='transit')
			for i in marks:
				ax0.add_line(lines.Line2D( [i,i],[min(flatten_lc1),max(flatten_lc1)],linestyle='dashed',color='red', label='>30 d gap'))
			ax0.set_xlabel( 'time [d] (Not showing time gaps > 30 d)')
			ax0.set_ylabel('flux')
			ax0.legend(loc='upper right')
			ax0.set_xlim([np.min(lc_time_clean),np.max(lc_time_clean)])
			ax0.set_ylim([np.min(flatten_lc1),np.mean(flatten_lc1)+np.std(flatten_lc1)*3])

			# Plotting the Periodgram
			ax2 = plt.subplot2grid((2,4),(0,2),colspan=2)
			ax2.set_title('Periodgram')
			ax2.plot(periods, power, 'k-', lw=2)
			ax2.add_line(lines.Line2D( [bls_results['bestperiod'],bls_results['bestperiod']],[min(power),max(power)],linestyle='dashed',color='red', label='best peak'))
			ax2.legend(loc='upper right')
			ax2.ticklabel_format(style='sci', axis='y', scilimits=(1,0))
			ax2.set_xlabel( 'period [days]' )
			ax2.set_xlim([min(periods), max(periods)])
			ax2.set_ylabel('BLS power')
			ax2.set_xscale('log')

			# Plotting the Phased Folded Lightcurve
			ax1 = plt.subplot2grid((2,4),(1,0),colspan=2)
			ax1.set_title('Phase Lightcurve')
			ax1.plot(phased_results['phase'], phased_results['mags'], '.k', label='phase folded')
			ax1.plot(binned_results['binnedphases'],binned_results['binnedmags'], '.', color="orange" ,label='binned')
			ax1.plot(phases, bls, color='red', label='model', linewidth=1.5)
			ax1.set_xlabel('phase')
			ax1.set_ylabel('flux')
			ax1.set_xlim([-0.2,0.8])
			ax1.set_ylim([np.min(phased_results['mags']),np.mean(phased_results['mags'])+np.std(phased_results['mags'])*3])
			ax1.legend(loc='upper right')
			ax1.plot()

			# Plotting transit zoom
			ax3 = plt.subplot2grid((2,4),(1,2),colspan=1)
			ax3.set_title('Zoom on binned phase folded transit')
			#ax3.plot(phased_results['phase'], phased_results['mags'], '.k', label='phase folded')
			ax3.plot(binned_results['binnedphases'],binned_results['binnedmags'], 'o', color="orange", label='binned')
			ax3.plot(phases, bls, color='red', label='model', linewidth=1.5)
			ax3.set_xlabel( 'phase')
			ax3.set_ylabel('flux')
			ax3.set_xlim([-tdur*5,tdur*5])
			if len(binned_results['binnedmags'])>0:
				ax3.set_ylim([np.amin(binned_results['binnedmags'])-np.std(binned_results['binnedmags']), np.amax(binned_results['binnedmags'])+np.std(binned_results['binnedmags'])])
			ax3.add_line(lines.Line2D( [0,0],[min(flatten_lc1),max(flatten_lc1)],linestyle='dashed',color='green', label='t0'))
			ax3.legend(loc='lower right')
			
			# Plotting secondary transit
			ax4 = plt.subplot2grid((2,4),(1,3),colspan=1)
			ax4.set_title('Zoom on secondary transit')
			#ax4.plot(phased_results['phase'], phased_results['mags'], '.k', label='phase folded')
			ax4.plot(binned_results['binnedphases'],binned_results['binnedmags'], 'o', color="orange", label='binned')
			ax4.set_xlabel( 'phase')
			ax4.set_ylabel('flux')
			ax4.set_xlim([0.5-tdur*5,0.5+tdur*5])
			if len(binned_results['binnedmags'])>0:
				ax4.set_ylim([np.amin(binned_results['binnedmags'])-np.std(binned_results['binnedmags']), np.amax(binned_results['binnedmags'])+np.std(binned_results['binnedmags'])])
			#ax4.set_xticks([0.4, 0.45, 0.5, 0.55, 0.6])
			ax4.legend(loc='lower right')
			

			# Subplot adjustment
			plt.subplots_adjust(left =.07, right=.99, hspace = .3, \
		                    wspace=.35, bottom = .15, top=.90)

			x0, y0 = ax1.get_position().x0, ax1.get_position().y0
			
			# Plotting statistics for this search
			fig.text(x0,y0-.12,'Period [fit value]: ' + str(round(p,5)) + '\nPeriod [bls]: ' + str(round(bls_results['bestperiod'],5)) + '\nFit Status : ' + str(stats['fit_status']),fontsize=15)
			fig.text(x0+.15,y0-.12,'Duration[phase]: ' + str(round(tdur,5)) + '\nDuration[hours]: ' + str(round((tdur*p)*24,5)) + '\nDepth: ' + str(round(depth,5)), fontsize=15)
			fig.text(x0+.3,y0-.1,'Epoch: ' + str(round(epoch0,5)) + '\n$R_p/R_s$: ' + str(round(math.sqrt(abs(depth)),5)), fontsize=15)
			fig.text(x0+.40,y0-.12,'SDE: ' + str(round(SDE,5)) + '\nSNR : ' + str(round(stats['snr'],5)) + '\nExposure Time[s]: ' + str(expt), fontsize=15)

			x0, y0 = ax0.get_position().x0, ax0.get_position().y0
			fig.text(x0+.40,y0+.35,lc_targ.TIC , fontsize=20)
			fig.text(x0,y0+.35,'Search number: ' + str(pt+1)  , fontsize=20)

		else:
			print('Negative transit depth for period trial #',str(pt+1),'in object',str(lc_targ.TIC), file=open("log.txt", "a"))
			break
		
		
		plt.savefig('im_'+str(pt)+'.png', dpi=400)
		plt.close(fig)
		

		if SDE<7 and pt<3:

			fig2 = plt.figure(figsize = (20,10))
		
			# Easy Handles
			stats = bls_results['stats'][1]
			epoch0 = stats['epoch']
			tdur = stats['transitduration'] #Transit Duration in phase units
			p = stats['period'] # Period in days
			bls = stats['blsmodel'] # BLS model values
			phases = stats['phases'] # Phases values
			depth = stats['transitdepth'] # Transit depth

			# Model wrap for plotting
			bls = np.concatenate((bls[np.where(phases>0.8)[0]],bls[np.where(np.logical_and(phases<=0.8, phases>=0))[0]]))			
			phases[np.where(phases>0.8)[0]] -= 1
			phases = np.sort(phases)

			# Astrobase methods
			# Phasing lightcurve
			phased_results = lcmath.phase_magseries_with_errs(lc_time, flatten_lc1, lc_err, p, epoch0, wrap=True, sort=True)

			# Binning phased lightcurve
			# Bin zise is 1/10 transit duration
			if tdur*0.1	< 0.001:
				bindur = 0.01
			else:
				bindur	= tdur
			binned_results = lcmath.phase_bin_magseries(phased_results['phase'], phased_results['mags'], binsize=bindur*0.1, minbinelems=7)

			# Intransit Masking
			intransit = transit_mask(lc_time, p, tdur*p, epoch0)

			# Plotting detrended lightcurve and marking transit~
			ax0 = plt.subplot2grid((2,4),(0,0),colspan=2)
			ax0.set_title('Detrended Lightcurve')
			lc_time_clean, marks = CleanTimeGap(lc_time,30, 5)
			ax0.plot(lc_time_clean[~intransit], flatten_lc1[~intransit],'k.', markersize=1.8, alpha = 0.25)
			ax0.plot(lc_time_clean[intransit], flatten_lc1[intransit],'o', color='orange', markersize=2, alpha = 0.5, label='transit')
			for i in marks:
				ax0.add_line(lines.Line2D( [i,i],[min(flatten_lc1),max(flatten_lc1)],linestyle='dashed',color='red', label='>30 d gap'))
			ax0.set_xlabel( 'time [d] (Not showing time gaps > 30 d)')
			ax0.set_ylabel('flux')
			ax0.legend(loc='upper right')
			ax0.set_xlim([min(lc_time_clean),max(lc_time_clean)])
			ax0.set_ylim([np.min(flatten_lc1),np.mean(flatten_lc1)+np.std(flatten_lc1)*3])

			# Plotting the Periodgram
			ax2 = plt.subplot2grid((2,4),(0,2),colspan=2)
			ax2.set_title('Periodgram')
			ax2.plot(periods, power, 'k-', lw=2) 
			ax2.add_line(lines.Line2D( [bls_results['nbestperiods'][1],bls_results['nbestperiods'][1]],[min(power),max(power)],linestyle='dashed',color='red', label='best peak'))
			ax2.legend(loc='upper right')
			ax2.ticklabel_format(style='sci', axis='y', scilimits=(1,0))
			ax2.set_xlabel('period [days]')
			ax2.set_xlim([min(periods), max(periods)])
			ax2.set_ylabel('BLS power')
			ax2.set_xscale('log')

			# Plotting the Phased Folded Lightcurve
			ax1 = plt.subplot2grid((2,4),(1,0),colspan=2)
			ax1.set_title('Phase Lightcurve')
			ax1.plot(phased_results['phase'], phased_results['mags'], '.k', label='phase folded')
			ax1.plot(binned_results['binnedphases'],binned_results['binnedmags'], '.', color="orange" , label='binned')
			ax1.plot(phases, bls, color='red', label='model', linewidth=1.5)
			ax1.set_xlabel('phase')
			ax1.set_ylabel('flux')
			ax1.set_xlim([-0.2,0.8])
			ax1.set_ylim([np.min(phased_results['mags']),np.mean(phased_results['mags'])+np.std(phased_results['mags'])*3])
			ax1.legend(loc='upper right')
			ax1.plot()

			# Plotting transit zoom
			ax3 = plt.subplot2grid((2,4),(1,2),colspan=1)
			ax3.set_title('Zoom on binned phase folded transit')
			#ax3.plot(phased_results['phase'], phased_results['mags'], '.k', label='phase folded')
			ax3.plot(binned_results['binnedphases'],binned_results['binnedmags'], 'o', color="orange" , label='binned')
			ax3.plot(phases, bls, color='red', label='model', linewidth=1.5)
			ax3.set_xlabel( 'phase')
			ax3.set_ylabel('flux')
			#ax3.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
			ax3.set_xlim([-tdur*5,tdur*5])
			if len(binned_results['binnedmags'])>0:
				ax3.set_ylim([np.amin(binned_results['binnedmags'])-np.std(binned_results['binnedmags']), np.amax(binned_results['binnedmags'])+np.std(binned_results['binnedmags'])])
			ax3.add_line(lines.Line2D( [0,0],[min(flatten_lc1[np.isfinite(flatten_lc1)]),max(flatten_lc1[np.isfinite(flatten_lc1)])],linestyle='dashed',color='green', label='t0'))
			ax3.legend(loc='lower right')

			# Plotting second transit
			ax4 = plt.subplot2grid((2,4),(1,3),colspan=1)
			ax4.set_title('Zoom on secondary transit')
			#ax4.plot(phased_results['phase'], phased_results['mags'], '.k', label='phase folded')
			ax4.plot(binned_results['binnedphases'],binned_results['binnedmags'], 'o', color="orange" , label='binned')
			ax4.set_xlabel( 'phase')
			ax4.set_ylabel('flux')
			ax4.set_xlim([0.5-tdur*5,0.5+tdur*5])
			if len(binned_results['binnedmags'])>0:
				ax4.set_ylim([np.amin(binned_results['binnedmags'])-np.std(binned_results['binnedmags']), np.amax(binned_results['binnedmags'])+np.std(binned_results['binnedmags'])])
			#ax4.set_xticks([0.4, 0.45, 0.5, 0.55, 0.6])
			ax4.legend(loc='lower right') 

			# Subplot adjustment
			plt.subplots_adjust(left =.07, right=.99, hspace = .3, \
			  wspace=.35, bottom = .15, top=.90)

			x0, y0 = ax1.get_position().x0, ax1.get_position().y0

			fig2.text(x0,y0-.12,'Period[fit value]: ' + str(round(p,5)) + '\nPeriod[bls]: ' + str(round(bls_results['nbestperiods'][1],5)) + '\nFit Status : ' + str(stats['fit_status']),fontsize=15)
			fig2.text(x0+.15,y0-.12,'Duration[phase]: ' + str(round(tdur,5)) + '\nDuration[hours]: ' + str(round((tdur*p)*24,5)) + '\nDepth: ' + str(round(depth,5)), fontsize=15)
			fig2.text(x0+.3,y0-.1,'Epoch: ' + str(round(epoch0,5)) + '\n$R_p/R_s$: ' + str(round(math.sqrt(abs(depth)),5)), fontsize=15)
			fig2.text(x0+.40,y0-.12,'SDE: ' + str(round(SDE,5)) + '\nSNR : ' + str(round(stats['snr'],5)) + '\nExposure Time[s]: ' + str(expt), fontsize=15)

			x0, y0 = ax0.get_position().x0, ax0.get_position().y0
			fig2.text(x0+.40,y0+.35,lc_targ.TIC , fontsize=20)
			fig2.text(x0,y0+.35,'Second peak of search number: ' + str(pt+1), fontsize=20)
			
			plt.savefig('im_'+str(pt+1)+'.png', dpi=400)
			plt.close(fig)
			
			print(tgt,'search Nº',pt+1,'SDE<7', file=open("log.txt", "a"))
			break

		# Taking transit out for next iteration
		flatten_lc1 = flatten_lc1[~intransit]
		lc_time = lc_time[~intransit]
		lc_err = lc_err[~intransit]			
				
	PDFCompiler(lc_targ.TIC, SDEmax, search)
	# Removing temporary PNG files
	for file in os.listdir(os.getcwd()):
		if file.endswith('.png'):
			os.remove(file) 
	print(tgt,' complete\n', file=open("log.txt", "a"))
	


