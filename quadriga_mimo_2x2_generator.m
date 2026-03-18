%% quadriga_mimo_2x2_generator.m
% Exercise 2.4(a)
% Generate a 2x2 MIMO flat-fading channel dataset with QuaDRiGa.
%
% Output variables saved in quadriga_mimo_2x2_dataset.mat
%   H_coeff : [Nr, Nt, Npaths, Nsnapshots] raw path coefficients
%   H_flat  : [Nr, Nt, Nsnapshots] flat-fading coefficients after path summation
%   H_vec   : [Nsnapshots, 2*Nr*Nt] real/imag interleaved vectorized channel snapshots
%
% Tested as MATLAB-style script; place it in the same folder as the QuaDRiGa library.

clear; close all; clc;

%% 0) Add QuaDRiGa to MATLAB path
% Adjust this path if your QuaDRiGa folder is elsewhere.
addpath(genpath("C:\Users\iwill\Downloads\QuaDRiGa-main\QuaDRiGa-main\quadriga_src"));

%% 1) Simulation parameters
fc              = 3.5e9;     % carrier frequency [Hz]
Nt              = 2;         % number of Tx antennas
Nr              = 2;         % number of Rx antennas
bs_height       = 25;        % base-station height [m]
ue_height       = 1.5;       % user height [m]
track_length_m  = 20;        % user track length [m]
samp_m          = 50;      % sample per meter
ue_speed_kmh    = 3;         % user speed [km/h]
ue_speed_mps    = ue_speed_kmh * 1000 / 3600;
lambda          = 3e8 / fc;
d               = 0.5 * lambda;  % half-wavelength spacing

%% 2) Create QuaDRiGa simulation object
s = qd_simulation_parameters;
s.center_frequency   = fc;
s.use_absolute_delays = 1;
s.show_progress_bars = 1;

l = qd_layout(s);
l.no_tx = 1;
l.no_rx = 1;

%% 3) Define 2x2 MIMO antenna arrays (simple ULA with omni elements)
% Tx array
l.tx_array = qd_arrayant('omni');
l.tx_array.copy_element(1, 2:Nt);
l.tx_array.element_position = [ ...
    zeros(1,Nt); ...
    linspace(-d/2, d/2, Nt); ...
    zeros(1,Nt) ...
];

% Rx array
l.rx_array = qd_arrayant('omni');
l.rx_array.copy_element(1, 2:Nr);
l.rx_array.element_position = [ ...
    zeros(1,Nr); ...
    linspace(-d/2, d/2, Nr); ...
    zeros(1,Nr) ...
];

%% 4) Tx/Rx geometry and track
l.tx_position = [0; 0; bs_height];
track = qd_track('linear', track_length_m, 0);  % move along +x direction
track.initial_position = [100; 0; ue_height];
track.interpolate_positions(samp_m);            % one sample every step_m meters
track.set_speed(ue_speed_mps);
l.rx_track = track;

num_snapshots = numel(track.positions(1,:));
fprintf('Snapshots = %d\n', num_snapshots);

%% 5) Scenario: 3GPP 38.901 Urban Micro, NLOS
l.set_scenario('3GPP_38.901_UMi_NLOS');
fprintf('Scenario = 3GPP_38.901_UMi_NLOS\n');
fprintf('Configuration = %dx%d MIMO, fc = %.1f GHz, UE speed = %.1f km/h\n', ...
    Nr, Nt, fc/1e9, ue_speed_kmh);

%% 6) Generate channel coefficients
b = l.init_builder;
gen_parameters(b);
c = get_channels(b);

% c.coeff dimensions: [Nr, Nt, Npaths, Nsnapshots]
H_coeff = c.coeff;

% Convert to flat-fading by summing over paths/clusters
H_flat = squeeze(sum(H_coeff, 3));   % [Nr, Nt, Nsnapshots]

%% 7) Vectorize each complex 2x2 matrix for GAN input
% Use real/imag interleaving:
% [Re(H11), Im(H11), Re(H21), Im(H21), Re(H12), Im(H12), Re(H22), Im(H22)]
% Note: MATLAB linear indexing stacks columns first.
H_vec = zeros(num_snapshots, 2 * Nr * Nt);
for k = 1:num_snapshots
    hk = H_flat(:,:,k);
    hk_col = hk(:);   % column-major vectorization
    tmp = zeros(1, 2*numel(hk_col));
    tmp(1:2:end) = real(hk_col).';
    tmp(2:2:end) = imag(hk_col).';
    H_vec(k,:) = tmp;
end

%% 8) Save dataset
save('quadriga_mimo_2x2_dataset.mat', 'H_coeff', 'H_flat', 'H_vec', ...
    'fc', 'Nt', 'Nr', 'bs_height', 'ue_height', 'track_length_m', ...
    'samp_m', 'ue_speed_kmh', '-v7');

fprintf('Saved dataset to quadriga_mimo_2x2_dataset.mat\n');
