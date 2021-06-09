src_folder = '/Users/mariamma/Documents/phd/rnn_project/code/dataset/blizzard_wavs_and_scores_2008_release_version_1/';
[X_train, X_test] = read_wav_files(src_folder);
writematrix(X_train,'/Users/mariamma/Documents/phd/rnn_project/code/dataset/X_train_blizzard.csv');
writematrix(X_test,'/Users/mariamma/Documents/phd/rnn_project/code/dataset/X_test_blizzard.csv');
    

function [X_train, X_test] = read_wav_files(src_folder)
    
    X_train = zeros(393120,200);
    X_test = zeros(62400,200);
    train_index = 0;
    test_index = 0;
    frame_overlap = 0.01;
    dim = 1;
    window_size = 0.01245;
    
    for alph=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',... 
            'M', 'N', 'O', 'P', 'Q', 'R','S', 'T', 'V']
        for num={'0002','0004','0009','0010','0011','0016','0021','0023',...
                '0024','0032','0037','0042','0044','0046','0050','0058',...
                '0061','0065','0066','0072','0078','0080','0081','0082'}
            filename = strcat(src_folder,alph,'/submission_directory/english/full/2008/news/news_2008_',num{1},'.wav');
            disp(filename)
            [x_original, Fs] = audioread(filename);
            calculated_frames_cnt = ((size(x_original,dim) - Fs*window_size)/(Fs * frame_overlap)) + 1;
            F = floor(calculated_frames_cnt);
            X_train(train_index + 1:train_index + F,:) = get_features_for_wav(x_original, Fs, F);
            train_index = train_index + F;
        end
        for num={'0090','0092','0093','0099'}
            filename = strcat(src_folder,alph,'/submission_directory/english/full/2008/news/news_2008_',num{1},'.wav');
            [x_original, Fs] = audioread(filename);
            calculated_frames_cnt = ((size(x_original,dim) - Fs*window_size)/(Fs * frame_overlap)) + 1;
            F = floor(calculated_frames_cnt);
            X_test(test_index + 1:test_index + F,:) = get_features_for_wav(x_original, Fs, F);
            test_index = test_index + F;
        end
        
    end
    
end


function [feat_mat] = get_features_for_wav(x_original, Fs, frames_cnt)
    
    x_normal = normalize(x_original);
    feat_mat = zeros(frames_cnt,200);
    dt = 1/Fs;
    dim = 1;
    window_size = 0.01245;
    last_frame_start = (size(x_original,dim) - Fs*window_size)/Fs;
    count = 1;
    for i = 0:.01:last_frame_start
    %     disp(i);
        I0 = round(i/dt);
        Iend = I0 + 200;
%         Iend = round((i + window_size)/dt);
    %     disp(['I0 : ', num2str(I0), ' Iend : ', num2str(Iend)])
%         if (I0 == 0)
%             I0 = 1;
%         end    
    %     disp(['Size of x ', num2str(size(x))])
        feat_mat(count,:) =  x_normal(I0+1:Iend).';
        count = count + 1;
    end
    disp(['Frames cnt :  ', num2str(count)]);
end

