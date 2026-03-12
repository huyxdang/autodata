Improvements

(**) Improvements on Prompt: 

❯ you should have tried more interesting approaches, like curriculum learning                                                                                                                    

⏺ You're right — I was stuck in a filtering mindset when the more interesting lever is how the model sees data, not just what data it sees. Curriculum learning, data mixing strategies, and
  ordering are likely where the bigger wins are.
                                                                                                                                                                                                 
  Want me to resume the experiment loop? I'll focus on more structural ideas:                                                                                                                    
                                                                                                                                                                                                 
  - Curriculum learning: shorter docs first, gradually increasing length as training progresses                                                                                                  
  - Difficulty-based ordering: easier (more common/simple) text early, harder text later
  - Epoch-aware mixing: vary the data composition as training progresses                                                                                                                         
  - Document shuffling within batches: ensuring each batch has diverse content                                                                                                                   
  - Upsampling high-quality docs: show the best documents multiple times                                                                                                                         
                                                                               
-------

(**) To come back to the original document: 


❯ but like, i won't know what the original data.py looks like.                                                                                                                                   
                                                                                                                                                                                                 
⏺ Good point — the agent kept overwriting data.py with each experiment. But you can always see the original via git:                                                                             
                                                                                                                                                                                                 
  git show 23fd775:data.py                                                                                                                                                                       
                                                                                                                                                                                                 
  That's the baseline commit. Or you can check the full history of changes to data.py:                                                                                                           
                                                                                                                                                                                                 
  git log --oneline -- data.py                                                                                                                                                                   



