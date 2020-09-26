In [19]: for i in range(len(orig_sub)):
    ...:     item = orig_sub.iloc[i]
    ...:     name = item['Audio_Name']
    ...:     pred_id = item['ID']
    ...:     
    ...:     bitrate = dict_bitrates[name]
    ...:     if bitrate == ' 256k':
    ...:         truth_range = speaker_256k        
    ...:         if not pred_id in truth_range:    
    ...:             print('256k: ',name,pred_id)  
    ...:     if bitrate == ' 768k':
    ...:         truth_range = speaker_768k 
    ...:         if not pred_id in truth_range:
    ...:             print('768k: ',name,pred_id)

