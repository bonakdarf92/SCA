% Number of multicasting subgroups
L = 2;

% Number of transmitt antennas
N = [3 5 8];

% Number of receivers 
M = 2:2:40;

% setup symbols
mod_order = 4;
mod = comm.PSKModulator('ModulationOrder', mod_order, 'PhaseOffset',-pi/4);
const = mod.constellation';
% information symbol
%symbol_ind = ceil(mod_order*rand(M,1));
%s = [];
for m = 1:length(M)
for l = 1:L
    s(m,l) = ceil(mod_order*rand(M(m),1));
end
end



