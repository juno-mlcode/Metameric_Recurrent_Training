%% gaussian
Ndat = 10000;


data = [0.1*randn(Ndat,1), 0.1*randn(Ndat,1); 1+ 0.1*randn(Ndat,1),0.1*randn(Ndat,1);0.1*randn(Ndat,1),1+0.1*randn(Ndat,1);1+0.1*randn(Ndat,1),1+0.1*randn(Ndat,1)];
fewdata =[0.5+0.1*randn(10,1), 0.5+0.1*randn(10,1)];
fewlabel = repmat([0 0 0 0 1],10,1);

data = [data, ones(4*Ndat,100)];
fewdata = [fewdata, -0.5*ones(10,100)];

traindata = [data;fewdata];
trainlabel = [repmat([1 0 0 0 0],Ndat,1);repmat([0 1 0 0 0],Ndat,1);repmat([0 0 1 0 0],Ndat,1);repmat([0 0 0 1 0],Ndat,1);repmat([0 0 0 0 1],10,1)];

Ndat = 1000;
testdata = [0.1*randn(Ndat,1), 0.1*randn(Ndat,1); 1+ 0.1*randn(Ndat,1),0.1*randn(Ndat,1);0.1*randn(Ndat,1),1+0.1*randn(Ndat,1);1+0.1*randn(Ndat,1),1+0.1*randn(Ndat,1)];
testlabel = [repmat([1 0 0 0 0],Ndat,1);repmat([0 1 0 0 0],Ndat,1);repmat([0 0 1 0 0],Ndat,1);repmat([0 0 0 1 0],Ndat,1)];
testdata1 = [0.5+0.1*randn(Ndat,1), 0.5+0.1*randn(Ndat,1)];
testlabel1 = repmat([0 0 0 0 1],Ndat,1);

testdata = [testdata, -0.5*ones(4*Ndat,100)];
testdata1 = [testdata1, -0.5*ones(Ndat,100)];

%scatter(traindata(:,1),traindata(:,2));

a = find(traindata<-0.5); b = find(traindata>1.5);
traindata(a) = -0.5; traindata(b) = 1.5;

a = find(testdata<-0.5); b = find(testdata>1.5);
traindata(a) = -0.5; traindata(b) = 1.5;

a = find(testdata1<-0.5); b = find(testdata1>1.5);
traindata(a) = -0.5; traindata(b) = 1.5;

traindata = (traindata+0.5)./2;
fewdata = (fewdata+0.5)./2;
testdata = (testdata+0.5)./2;
testdata1 = (testdata1+0.5)./2;

hold on;
scatter(traindata(:,1),traindata(:,2));
scatter(testdata1(:,1),testdata1(:,2));



%% ±âµÕ ¿Ü°û
% 
% Ndat = 10000;
% data = [rand(Ndat,1),rand(Ndat,1); 1.5+rand(Ndat,1),rand(Ndat,1);3+rand(Ndat,1),rand(Ndat,1);4.5+rand(Ndat,1),rand(Ndat,1)];
% fewdata = [6+rand(10,1),rand(10,1)]; 
% fewlabel = repmat([0 0 0 0 1],10,1);
% traindata = [data;fewdata];
% trainlabel = [repmat([1 0 0 0 0],Ndat,1);repmat([0 1 0 0 0],Ndat,1);repmat([0 0 1 0 0],Ndat,1);repmat([0 0 0 1 0],Ndat,1);repmat([0 0 0 0 1],10,1)];
% 
% Ndat = 1000;
% testdata = [rand(Ndat,1),rand(Ndat,1); 1.5+rand(Ndat,1),rand(Ndat,1);3+rand(Ndat,1),rand(Ndat,1);4.5+rand(Ndat,1),rand(Ndat,1)];
% testlabel = [repmat([1 0 0 0 0],Ndat,1);repmat([0 1 0 0 0],Ndat,1);repmat([0 0 1 0 0],Ndat,1);repmat([0 0 0 1 0],Ndat,1)];
% 
% testdata1 = [6+rand(Ndat,1),rand(Ndat,1)];
% testlabel1 = repmat([0 0 0 0 1],Ndat,1);
% %scatter(traindata(:,1),traindata(:,2));
% 
% traindata(:,1) = traindata(:,1)/7;
% testdata(:,1) = testdata(:,1)/7;
% testdata1(:,1) = testdata1(:,1)/7;
% fewdata(:,1) = fewdata(:,1)/7;
% hold on;
% scatter(traindata(:,1),traindata(:,2));
% scatter(testdata1(:,1),testdata1(:,2));
% 
% 
% testdata2 = zeros(101*101,2);
% for i =0:0.01:1
%     for j=0:0.01:1
%         testdata2(round(101*i*100+j*100+1),:) = [i,j];
%     end
% end

%% ±âµÕ °¡¿îµ¥
% 
% Ndat = 10000;
% data = [rand(Ndat,1),rand(Ndat,1); 1.5+rand(Ndat,1),rand(Ndat,1);4.5+rand(Ndat,1),rand(Ndat,1);6+rand(Ndat,1),rand(Ndat,1)];
% fewdata = [3+rand(10,1),rand(10,1)]; 
% fewlabel = repmat([0 0 0 0 1],10,1);
% traindata = [data;fewdata];
% trainlabel = [repmat([1 0 0 0 0],Ndat,1);repmat([0 1 0 0 0],Ndat,1);repmat([0 0 1 0 0],Ndat,1);repmat([0 0 0 1 0],Ndat,1);repmat([0 0 0 0 1],10,1)];
% 
% Ndat = 1000;
% testdata = [rand(Ndat,1),rand(Ndat,1); 1.5+rand(Ndat,1),rand(Ndat,1);4.5+rand(Ndat,1),rand(Ndat,1);6+rand(Ndat,1),rand(Ndat,1)];
% testlabel = [repmat([1 0 0 0 0],Ndat,1);repmat([0 1 0 0 0],Ndat,1);repmat([0 0 1 0 0],Ndat,1);repmat([0 0 0 1 0],Ndat,1)];
% 
% testdata1 = [3+rand(Ndat,1),rand(Ndat,1)];
% testlabel1 = repmat([0 0 0 0 1],Ndat,1);
% %scatter(traindata(:,1),traindata(:,2));
% 
% traindata(:,1) = traindata(:,1)/7;
% testdata(:,1) = testdata(:,1)/7;
% testdata1(:,1) = testdata1(:,1)/7;
% fewdata(:,1) = fewdata(:,1)/7;
% hold on;
% scatter(traindata(:,1),traindata(:,2));
% scatter(testdata1(:,1),testdata1(:,2));
% % 
% % 
% % testdata2 = zeros(101*101,2);
% % for i =0:0.01:1
% %     for j=0:0.01:1
% %         testdata2(round(101*i*100+j*100+1),:) = [i,j];
% %     end
% % end
% 
% 
