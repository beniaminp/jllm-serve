package org.robomiri.jllm.server;

/** A minimal single-file chat page served at {@code /} that talks to the streaming chat endpoint. */
final class WebUi {

    private WebUi() {
    }

    static final String HTML = """
            <!doctype html>
            <html><head><meta charset="utf-8"><title>jllm-serve</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
              body{font-family:system-ui,sans-serif;max-width:800px;margin:0 auto;padding:16px;background:#111;color:#eee}
              #log{white-space:pre-wrap;border:1px solid #333;border-radius:8px;padding:12px;min-height:300px;margin-bottom:12px}
              .u{color:#8cf}.a{color:#eee}.s{color:#888;font-size:12px}
              form{display:flex;gap:8px}input{flex:1;padding:10px;border-radius:8px;border:1px solid #444;background:#222;color:#eee}
              button{padding:10px 16px;border-radius:8px;border:0;background:#4a7;color:#fff}
              label{font-size:13px;color:#aaa}
            </style></head><body>
            <h2>jllm-serve</h2>
            <div id="log"></div>
            <form id="f"><input id="q" autocomplete="off" placeholder="Say something..."><button>Send</button></form>
            <label><input type="checkbox" id="think"> enable thinking</label>
            <script>
            const log=document.getElementById('log'),msgs=[];
            document.getElementById('f').onsubmit=async e=>{
              e.preventDefault();const q=document.getElementById('q');const text=q.value.trim();if(!text)return;q.value='';
              msgs.push({role:'user',content:text});
              log.insertAdjacentHTML('beforeend','<div class="u">you: '+esc(text)+'</div>');
              const a=document.createElement('div');a.className='a';a.textContent='assistant: ';log.appendChild(a);
              let full='',usage=null,err=null;
              try{
                const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
                  body:JSON.stringify({messages:msgs,stream:true,enable_thinking:document.getElementById('think').checked})});
                if(!r.ok){err='HTTP '+r.status;try{const j=await r.json();if(j.error?.message)err=j.error.message;}catch(_){}}
                else{
                  const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
                  read:for(;;){const{value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
                    let i;while((i=buf.indexOf('\\n\\n'))>=0){const line=buf.slice(0,i).trim();buf=buf.slice(i+2);
                      if(!line.startsWith('data: '))continue;const d=line.slice(6);if(d==='[DONE]')continue;
                      const j=JSON.parse(d);if(j.error){err=j.error.message||'error';break read;}
                      const c=j.choices?.[0]?.delta?.content;if(c){full+=c;a.textContent='assistant: '+full;}
                      if(j.usage)usage=j.usage;window.scrollTo(0,document.body.scrollHeight);}}
                }
              }catch(e){err=String(e);}
              // keep the history consistent: no empty assistant turn, and a failed user turn is not re-sent
              if(full)msgs.push({role:'assistant',content:full});else{msgs.pop();a.remove();}
              if(err)log.insertAdjacentHTML('beforeend','<div class="s">error: '+esc(err)+'</div>');
              if(usage)log.insertAdjacentHTML('beforeend','<div class="s">'+usage.completion_tokens+' tokens, '+usage.tokens_per_second+' tok/s, prefill '+usage.prefill_ms+' ms</div>');
              window.scrollTo(0,document.body.scrollHeight);
            };
            function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
            </script></body></html>
            """;
}
