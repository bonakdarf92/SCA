using Convex
using SCS

PLOT_FIGURES = true # True to plot figures, false to suppress plots
N = 96 # Number of periods in the day (so each interval 15 minutes)

# Convenience variables for plotting
fig_size = [14,3];
xtick_vals = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96];
xtick_labels = ["0:00", "2:00am", "4:00am", "5:00am", "8:00am", "10:00am", "12:00pm", "2:00pm", "4:00pm", "6:00pm", "8:00pm", "10:00pm", "12:00am"];


#############################################
# Price data generation - price values and intervals based off of PG&E Time Of Use plans
#############################################
partial_peak_start = 34   # 08:30
peak_start = 48           # 12:00
peak_end = 72             # 18:00 (6:00pm)
partial_peak_end = 86     # 21:30 (9:30pm)

off_peak_inds = vcat(1:partial_peak_start, partial_peak_end+1:N);
partial_peak_inds = vcat(partial_peak_start+1:peak_start, peak_end+1:partial_peak_end);
peak_inds = vcat(peak_start+1:peak_end);

# rates in $ / kWh
off_peak_buy = 0.14;
partial_peak_buy = 0.25;
peak_buy = 0.45;

# Rate cuts from buy prices to get sell prices
off_peak_perc_cut = 0.20;
partial_peak_perc_cut = 0.12;
peak_perc_cut = 0.11;


off_peak_sell = (1 - off_peak_perc_cut) * off_peak_buy;
partial_peak_sell = (1 - partial_peak_perc_cut) * partial_peak_buy;
peak_sell = (1 - peak_perc_cut) * peak_buy;

# Combine the buy and sell prices into the price vectors
R_buy = zeros(N,1);
R_buy[off_peak_inds] .= off_peak_buy;
R_buy[partial_peak_inds] .= partial_peak_buy;
R_buy[peak_inds] .= peak_buy;

R_sell = zeros(N,1);
R_sell[off_peak_inds] .= off_peak_sell;
R_sell[partial_peak_inds] .= partial_peak_sell;
R_sell[peak_inds] .= peak_sell;
