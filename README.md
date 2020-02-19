# RNN_NMF_chatbot
Topic-aware chatbot based on RNN seq2seq model and NMF-based topic baising

## References

These codes are based on my papers below: 
  1. Yuchen Guo, Nicholas Hanoian, Zhexiao Lin, Nicholas Liskij, Hanbaek Lyu, Deanna Needell, Jiahao Qu, Henry Sojico, Yuliang Wang, Zhe Xiong, Zhenhong Zou, 
     “Topic-aware chatbot using Recurrent Neural Networks and Nonnegative Matrix Factorization.” 
     https://arxiv.org/abs/1912.00315

## File description 

  1. **seq2seq.py** : Basic RNN seq2seq model for chatbot  
  2. **ta_seq2seq.py** : Gives TopicAttension and TopicDecoder -- RNN decoder for generating predicted probability distirbution for the next word using NMF-induced topic biasing
  3. **seq2seq_chatbot_train** : Trains chatbot over the given conversaional data set and chosen NMF-topic filter (DeltaAirline, 20NewsGroups, and Shakespeare)
  4. **chat_app.py** : Run and chat with chosen topic-aware chatbot model
  
## Authors

* **Yuliang Wang** - *Initial work* 
* **Hanbaek Lyu** - *Initial work* - [Website](https://hanbaeklyu.com)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Initial seq2seq chatbot model in pytorch -- Matthew Inkawhich [Link](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
* Project has been supported by REU 2019 at UCLA Mathematics
