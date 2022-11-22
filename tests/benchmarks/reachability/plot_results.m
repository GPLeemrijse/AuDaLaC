plot3(results.n, results.m, results.timems, 'x');
title("Logarithm of graph dimensions vs logarithm of runtime.")
xlabel('n')
ylabel('m')
zlabel('time (ms)')
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
set(gca, 'ZScale', 'log')
grid on
%% 

figure
plot3(results.n, results.m, results.timems ./ (results.n .* results.m), 'x');
title("Logarithm of graph dimensions vs logarithm of runtime per data element.")
xlabel('n')
ylabel('m')
zlabel('time / problemsize')
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
set(gca, 'ZScale', 'log')
grid on
%% 

figure
scatter(results.m, results.timems ./ (results.n .* results.m))
xlabel('m')
ylabel('time / problemsize')
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
grid on

%% 

figure
scatter(results.n, results.timems ./ (results.n .* results.m))
xlabel('n')
ylabel('time / problemsize')
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
grid on
