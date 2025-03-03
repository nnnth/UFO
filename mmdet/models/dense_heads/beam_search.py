import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty # 长度惩罚的指数系数
        self.num_beams = num_beams # beam size
        self.beams = [] # 存储最优序列及其累加的log_prob score
        self.worst_score = 1e9 # 将worst_score初始为无穷大。

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs, his_scores, his_feats):
        score = sum_logprobs / len(hyp) ** self.length_penalty # 计算惩罚后的score
        if len(self) < self.num_beams or score > self.worst_score:
            # 如果类没装满num_beams个序列
            # 或者装满以后，但是待加入序列的score值大于类中的最小值
            # 则将该序列更新进类中，并淘汰之前类中最差的序列
            self.beams.append((score, hyp, his_scores, his_feats))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _, _, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                                # 如果没满的话，仅更新worst_score
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        # 当解码到某一层后, 该层每个结点的分数表示从根节点到这里的log_prob之和
        # 此时取最高的log_prob, 如果此时候选序列的最高分都比类中最低分还要低的话
        # 那就没必要继续解码下去了。此时完成对该句子的解码，类中有num_beams个最优序列。
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

# batch_size = 3
# num_beams = 2
# vocab_size = 8
# cur_len = 1
# embedding_size = 300
# hidden_size = 100
# max_length = 10
# sos_token_id = 0
# eos_token_id = 1
# pad_token_id = 2
# decoder = DecoderRNN(embedding_size, hidden_size, vocab_size)


def beam_search(model, input_ids, pre_kv_list, batch_size, num_beams,pre_len, max_length, global_only_image,temperature, length_penalty, bos_token_id,eos_token_id,pad_token_id):
    beam_scores = torch.zeros((batch_size, num_beams)).to(input_ids.device) # 定义scores向量，保存累加的log_probs
    beam_scores[:, 1:] = -1e9 # 需要初始化为-inf
    beam_scores = beam_scores.view(-1) # 展开为(batch_size * num_beams)
    done = [False for _ in range(batch_size)] # 标记每个输入句子的beam search是否完成
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty=length_penalty)
            for _ in range(batch_size)
    ] # 为每个输入句子定义维护其beam search序列的类实例
    # 初始输入: （batch_size * num_beams, 1）个sos token
    # input_ids =  torch.full((batch_size * num_beams, 1), sos_token_id, dtype=torch.long)
    # h0: (1, batch_size * num_beams, hidden_size)
    # hidden = torch.zeros((1, batch_size * num_beams, hidden_size))
    cur_len = 0 
    vocab_size = model.embed.word_embeddings.weight.shape[0]
    pre_kv_list = model.repeat_cache(num_beams, pre_kv_list)
    
    # record history scores and grid references
    his_scores = torch.zeros_like(input_ids).to(torch.float)
    his_feats = torch.zeros_like(input_ids).to(torch.float).unsqueeze(-1).repeat(1,1,model.embed_dims)
    
    while cur_len < max_length:
        # outputs: (batch_size*num_beams, cur_len, vocab_size)
        # TODO: references in window attention
        feats, outputs, pre_kv_list = model.decode_forward(input_ids[:,-1:], pre_kv_list, pre_len+cur_len, global_only_image)
        # 取最后一个timestep的输出 (batch_size*num_beams, vocab_size)
        next_token_logits = outputs[:, -1, :] / temperature
        scores = F.log_softmax(next_token_logits, dim=-1) # log_softmax

        next_scores = scores + beam_scores[:, None].expand_as(scores) # 累加上以前的scores
        next_scores = next_scores.view(
            batch_size, num_beams * vocab_size
        ) # 转成(batch_size, num_beams * vocab_size), 如上图所示
        # 取topk
        # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
        next_scores, next_tokens = torch.topk(next_scores, 2*num_beams, dim=1, largest=True, sorted=True)

        feats = feats.view(batch_size, num_beams, -1)
        next_beams = next_tokens // vocab_size
        next_feats = torch.gather(feats, 1, next_beams.unsqueeze(-1).repeat(1,1,model.embed_dims))
        
        next_batch_beam = []

        for batch_idx in range(batch_size):
            if done[batch_idx]:
                # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                next_batch_beam.extend([(0, pad_token_id, 0, torch.zeros_like(feats[0][0]))] * num_beams)  # pad the batch
                continue
            next_sent_beam = [] # 保存三元组(beam_token_score, token_id, effective_beam_id)
            for beam_token_rank, (beam_token_id, beam_token_score, beam_token_feat) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx], next_feats[batch_idx])
                ):
                beam_id = beam_token_id // vocab_size # 1
                token_id = beam_token_id % vocab_size # 1
                # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                # batch_idx=1时，真实beam_id如下式计算为4或5
                effective_beam_id = batch_idx * num_beams + beam_id
                # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score.item(), his_scores[effective_beam_id], his_feats[effective_beam_id]
                    )

                    # DEBUG
                    # next_sent_beam.append([0,token_id,effective_beam_id])
                    # done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    #         next_scores[batch_idx].max().item(), cur_len
                    #     ) # 注意这里取当前batch的所有log_prob的最大值
                else:
                    # Debug
                    # if input_ids[effective_beam_id,-1] == eos_token_id:
                    #     beam_token_score = 0 
                    #     token_id = eos_token_id
                    # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id, beam_token_feat))

                if len(next_sent_beam) == num_beams:
                    break
                # 当前batch是否解码完所有句子
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                            next_scores[batch_idx].max().item(), cur_len
                        ) # 注意这里取当前batch的所有log_prob的最大值
            # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
            # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
            next_batch_beam.extend(next_sent_beam)
        # 如果batch中每个句子的beam search都完成了，则停止
        if all(done):
            break
        # 准备下一次循环(下一层的解码)
        # beam_scores: (num_beams * batch_size)
        # beam_tokens: (num_beams * batch_size)
        # beam_idx: (num_beams * batch_size) 
        # 这里beam idx shape不一定为num_beams * batch_size，一般是小于等于
        # 因为有些beam id对应的句子已经解码完了 (下面假设都没解码完)
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        beam_feats = torch.stack([x[3] for x in next_batch_beam],dim=0)
        # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面, 
        # 因为有些beam id对应的句子已经解码完了
        input_ids = input_ids[beam_idx, :] # (num_beams * batch_size, seq_len)
        # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        
        his_scores = his_scores[beam_idx, :]
        his_scores = torch.cat([his_scores, beam_scores.unsqueeze(1)], dim=-1)
        
        his_feats = his_feats[beam_idx, :]
        his_feats = torch.cat([his_feats, beam_feats.unsqueeze(1)], dim=1)

        cur_len = cur_len + 1
        
        pre_kv_list = model.reorder_cache(beam_idx, pre_kv_list)
        
    # 注意有可能到达最大长度后，仍然有些句子没有遇到eos token，这时done[batch_idx]是false
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue
        for beam_id in range(num_beams):
            # 对于每个batch_idx的每句beam，都执行加入add
            # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            final_his_scores = his_scores[effective_beam_id]
            final_his_feats = his_feats[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score, final_his_scores, final_his_feats)
    # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
    # 下面选择若干最好的序列输出
    # 每个样本返回几个句子
    output_num_return_sequences_per_batch = 1
    output_batch_size = output_num_return_sequences_per_batch * batch_size
    # 记录每个返回句子的长度，用于后面pad
    sent_lengths = input_ids.new(output_batch_size)
    best = []
    best_scores = []
    best_feats = []
    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        # x: (score, hyp), x[0]: score
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp, best_s, best_f = sorted_hyps.pop()[1:]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)
            best_scores.append(best_s)
            best_feats.append(best_f)
            
    if sent_lengths.min().item() != sent_lengths.max().item():
        # Debug
        sent_max_len = min(sent_lengths.max().item() + 1, max_length+1)
        # fill pad
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)
        decoded_scores = input_ids.new(output_batch_size, sent_max_len).fill_(0).to(torch.float)
        decoded_feats = input_ids.new(output_batch_size, sent_max_len, model.embed_dims).fill_(0).to(torch.float)
 
        # 填充内容
        for i, hypo in enumerate(best):
            try:
                decoded[i, : sent_lengths[i]] = hypo
            except:
                import ipdb 
                ipdb.set_trace()
            decoded_scores[i, : sent_lengths[i]] = best_scores[i]
            decoded_feats[i, : sent_lengths[i]] = best_feats[i]
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
                decoded[i, sent_lengths[i]] = 0
    else:
        # 否则直接堆叠起来
        decoded = torch.stack(best).type(torch.long)
        decoded_scores = torch.stack(best_scores).type(torch.float)
        decoded_feats = torch.stack(best_feats).type(torch.float)
    decoded = decoded[:,1:]
    decoded_scores = (decoded_scores[:,1:] - decoded_scores[:,:-1]).exp()
    decoded_feats=  decoded_feats[:,1:]
    # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
    return decoded, decoded_scores, decoded_feats