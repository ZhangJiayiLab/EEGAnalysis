%Upstate detection
%by Jiayi Zhang, Jan 16 2014

%Input data: 0-0.5Hz Power spectrum below 300Hz
 
clear all;
clc;

power_spectrum_raw = xlsread('D:\新建文件夹\165.xlsx'); 
power_spectrum = power_spectrum_raw(2:end,:);
interval_threshold = 0.4;       % time difference between two events in sec
Time_difference = 1;
upstate_duration = 0.15;       % threshold of upstate duration in sec

Event_interval=zeros(200,size(power_spectrum_raw,2)-1);
for channelnum=1:size(power_spectrum_raw,2)-1
    power_value = power_spectrum(:,channelnum+1);
    t = func_threshold(power_value);
    Result = power_spectrum;
    Result(power_spectrum(:,2) < t) = 0;  %set the time below threshold to be 0


    a = Result(:,1);
    Result_time = a(a ~= 0);

    Mark_start(1,1) = Result_time(1,1);
    Mark_start(1,2) = 1;
    k = 1;

    for i = 1:size(Result_time)-1

        if(Result_time(i+1)-Result_time(i) > Time_difference)
            Mark_end(k,1) = Result_time(i);
            Mark_end(k,2) = i;
            Mark_start(k+1,1) = Result_time(i+1);
            Mark_start(k+1,2) = i+1;

            k = k+1;
        end

    end

    Mark_end(k,1) = max(Result_time);
    Mark_end(k,2) = 0;

    for j = 1:size(Mark_start,1)-10
        if (Mark_end(j,1) - Mark_start(j,1)) < upstate_duration
            Mark_start(j,:) = [];
            Mark_end(j,:) = [];
        end
    end

    Event_interval(1:size(Mark_start,1)-1,channelnum) = diff(Mark_start(:,1));

    Event_duration = Mark_end - Mark_start;
    
    dele=[];
    
    dele=find(Event_interval(:,channelnum) < interval_threshold);
    
    %(dele>)

    Event_interval(dele,channelnum) =0;

    Event_timestamp = cumsum(Event_interval);

    Mark_peak = (Mark_start + Mark_end)/2;


end
