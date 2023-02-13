%% 
results2 = sortrows(results, ["n", "m", "timems"]);

[ps_mask, p_sizes] = findgroups(results.problemsize);

for ps = p_sizes'
    p_set = results(results.problemsize == ps,:);
    if height(p_set) >= 6
        figure()
        p_set = sortrows(p_set, "m");
        plot(p_set.m, p_set.timems, '-x');
        title(sprintf('Width of graph vs wall-clock-time in ms\nfor a problem size of %d nodes', ps));
        xlabel("m");
        xlim([min(p_set.m), max(p_set.m)]);
        ylabel("time (ms)");
        set(gca, 'XScale', 'log')
        xticks(p_set.m);
        ticklabels = arrayfun(@(l) sprintf('2^{%d}', log2(l)), p_set.m, 'UniformOutput', false);
        xticklabels(ticklabels)
        %set(gca, 'YScale', 'log')
    end
end

%% 
figure()
hold on
large_m = 0;
small_m = 1000000000;
i = 1;
markers = {'+','o','*','x','h','s','d','^','v','>','<','p','.'};
used_ps = [];
for ps = p_sizes'
    p_set = results(results.problemsize == ps,:);
    if height(p_set) >= 6
        p_set = sortrows(p_set, "m");
        p = plot(p_set.m, p_set.timems);
        p.Marker = markers{mod(i,numel(markers))+1};
        small_m = min(small_m, min(p_set.m));
        large_m = max(large_m, max(p_set.m));
        i = i + 1;
        used_ps = [used_ps ps];
    end
end
xlabel("m");
ylabel("time (ms) / problemsize");
set(gca, 'XScale', 'log')
title(sprintf('Width of graph vs wall-clock-time in ms'));
xlim([small_m, large_m]);
tickx = arrayfun(@(x) pow2(x),log2(small_m):log2(large_m));
xticks(tickx);
ticklabels = arrayfun(@(l) sprintf('2^{%d}', log2(l)), tickx, 'UniformOutput', false);
xticklabels(ticklabels)
legend(arrayfun(@(l) sprintf('problemsize=%d', l), used_ps, 'UniformOutput', false));
hold off
%% 

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
%set(gca, 'ZScale', 'log')
grid on

%% 
results2 = results(not(and(results.n == 1, results.m == 2)),:);
figure
plot3(results2.n, results2.m, results2.timems ./ (results2.n .* results2.m), 'x');
title("Logarithm of graph dimensions vs logarithm of runtime per data element.")
xlabel('n')
ylabel('m')
zlabel('time / problemsize')
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
%set(gca, 'ZScale', 'log')
grid on
