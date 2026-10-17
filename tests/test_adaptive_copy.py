import json,unittest
from pathlib import Path
from tensorfold.families.glm5_next.cuda.copy_drafts import CopySettings
from tensorfold.families.glm5_next.cuda.adaptive_copy import AdaptiveCopyDrafts as Adaptive

def records():
 return [1000+i*32+j if j<10 else (99 if j==25 else j) for i in range(12) for j in range(36)]

class Checks(unittest.TestCase):
 def test_two_distinct_positions_and_conflict(self):
  p=[1,90,2,90,3,90];d=Adaptive(p+[7],prompt=len(p))
  self.assertFalse(d._observe_span(1,2,[91]));self.assertFalse(d._observe_span(1,2,[91]));self.assertFalse(d.edits)
  self.assertTrue(d._observe_span(3,4,[91]));self.assertEqual(d.edits,{(90,):(91,)})
  self.assertFalse(d._observe_span(5,6,[92]));self.assertFalse(d.edits);self.assertEqual(d.buf[:6].tolist(),p)
 def test_verified_variable_length_edits_and_ranks(self):
  p=records();expected=[]
  for i in range(12):
   block=p[i*36:(i+1)*36];expected+=block[:25]+[101,102]+block[29:]
  ds=[Adaptive(p+[expected[0]],prompt=len(p),most=15,reply_match=32) for _ in range(2)]
  at=1;rounds=0;copies=0
  while at<len(expected):
   ps=[d.propose(len(expected)-at) for d in ds];self.assertEqual(*ps)
   n=0
   while n<len(ps[0]) and at+n<len(expected) and ps[0][n]==expected[at+n]:n+=1
   size=min(n+1 if ps[0] else 8,len(expected)-at)
   for d in ds:d.extend(expected[at:at+size])
   self.assertEqual(ds[0].edits,ds[1].edits)
   at+=size;rounds+=1;copies+=n
  self.assertTrue(ds[0].edits);self.assertFalse(ds[0].disabled);self.assertLess(rounds,40)
  self.assertEqual(ds[0].original.tolist(),p)
 def test_ambiguous_continuations_abstain(self):
  suffix=list(range(10,26));p=[1]+suffix+[100,101,102,103]+[2]+suffix+[200,201,202,203]
  self.assertEqual(Adaptive(p+[3]+suffix,prompt=len(p),most=7).propose(),[])
 def test_rank_settings_and_invalid_values(self):
  a=CopySettings.from_env(15,{'TF_GLM_COPY_DRAFTS':'1','TF_GLM_COPY_SMART':'1'})
  b=CopySettings.from_env(15,{'TF_GLM_COPY_DRAFTS':'1','TF_GLM_COPY_SMART':'0'})
  self.assertNotEqual(a.code(),b.code())
  with self.assertRaises(ValueError):CopySettings.from_env(15,{'TF_GLM_COPY_DRAFTS':'1','TF_GLM_COPY_SMART':'2'})
 def test_room_and_neural_fallback(self):
  p=records();d=Adaptive(p+p[:25],prompt=len(p),most=15)
  self.assertEqual(d.propose(0),[]);self.assertLessEqual(len(d.propose(7)),7)
  d.edits[(99,)]=(101,);d._rebuild()
  for _ in range(3):d.last=[7,8,9];d.extend([7,10])
  self.assertTrue(d.disabled);self.assertEqual(d.propose(),[])
 def test_learned_deletion(self):
  p=[1,90,91,2,90,91];d=Adaptive(p+[7],prompt=len(p))
  self.assertFalse(d._observe_span(1,3,[]));self.assertTrue(d._observe_span(4,6,[]))
  self.assertEqual(d.buf[:d.prompt].tolist(),[1,2]);self.assertEqual(d.reply,[7])

 def test_wide_window_sizes_buffers(self):
  from tensorfold.families.glm5_next.cuda.engine import multi_window,multi_decode_rows
  self.assertEqual(multi_window('128'),128)
  self.assertEqual(multi_decode_rows(8,128),144)
  for invalid in ('0','129','136'):
   with self.assertRaises(ValueError):multi_window(invalid)
 def test_real_tokenizer_span_regression(self):
  j=json.loads((Path(__file__).parent/'fixtures/repeated-edit.json').read_text())
  p,o=j['prompt'],j['output'];d=Adaptive(p+[o[0]],prompt=len(p),most=15,reply_match=32)
  at=1
  while at<128:
   proposal=d.propose(128-at);n=0
   while n<len(proposal) and at+n<128 and proposal[n]==o[at+n]:n+=1
   size=min(n+1 if proposal else 8,128-at)
   d.extend(o[at:at+size]);at+=size
  self.assertTrue(d.edits);self.assertFalse(d.disabled)
  self.assertTrue(any(len(k)!=len(v) for k,v in d.edits.items()))

 def test_batched_initializer_uses_adaptive_factory(self):
  from types import SimpleNamespace as NS
  from tensorfold.families.glm5_next.cuda.multi import MultiDecoder
  d=MultiDecoder.__new__(MultiDecoder)
  c=CopySettings.from_env(15,{'TF_GLM_COPY_DRAFTS':'1','TF_GLM_COPY_SMART':'1'},pad=(16,))
  d.g=NS(copy=c)
  lane=NS(s=NS(prompt=list(range(30)),count=128),policy=NS(next=lambda *a:7),copy_enabled=True,copies=None)
  d._started(lane,42)
  self.assertIsInstance(lane.copies,Adaptive);self.assertEqual(lane.copies.pad,())
  lane.copy_enabled=False;lane.copies=None;d._started(lane,42)
  self.assertIsNone(lane.copies)
 def test_unused_proposal_does_not_count_as_failure(self):
  d=Adaptive([1,2,3,4,5],prompt=4)
  d.edits[(1,)]=(9,);d.last=[7,8];d.source=1
  d.discard();d.extend([10]);self.assertEqual(d.failures,0)
 def test_cached_head_priority_and_cold_fill_fairness(self):
  from types import SimpleNamespace as NS
  from unittest.mock import patch
  from tensorfold.families.glm5_next.cuda import multi
  d=multi.MultiDecoder.__new__(multi.MultiDecoder)
  cold=NS(decoding=False,head=None,order=0,s=NS(sid=0))
  hit=NS(decoding=False,head=object(),order=1,s=NS(sid=1))
  d.lanes={0:cold,1:hit};d.count=8;d.partial=None;d.fill_budget=0
  self.assertIs(d._fill_turn(),hit)
  d.partial=NS(group=[(cold,0)])
  self.assertIs(d._fill_turn(),cold)
  d.partial=None
  hit.order=100
  with patch.object(multi,'next_fill',return_value=cold.s):self.assertIs(d._fill_turn(),cold)
 def test_copy_option_is_transmitted(self):
  from tensorfold.families.glm5_next.cuda import multi
  for enabled in (False,True):
   p=[1,0,0,16,-1,0,128,1,1]+multi.pack_sampling(None)+[0]*multi.SHARED_SLOTS+[0]*4
   p+=[1,42,0,0,0]+multi.pack_keys([],[],[])+[0,int(enabled)]
   self.assertEqual(multi.MultiDecoder._parse_admit(p)['copy_enabled'],enabled)
 def test_invalid_copy_option_is_rejected(self):
  from tensorfold.families.glm5_next.cuda.app import GlmApp
  self.assertIn('boolean',GlmApp._check_fields(GlmApp.__new__(GlmApp),{'tf_copy':'yes'}))

if __name__=='__main__':unittest.main()
